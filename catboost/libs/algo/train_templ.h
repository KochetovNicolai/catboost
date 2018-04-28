#pragma once

#include "approx_calcer.h"
#include "fold.h"
#include "greedy_tensor_search.h"
#include "online_ctr.h"
#include "tensor_search_helpers.h"

#include <catboost/libs/distributed/worker.h>
#include <catboost/libs/distributed/master.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/logging/profile_info.h>
#include <iostream>

struct TCompetitor;

namespace {
void NormalizeLeafValues(const TVector<TIndexType>& indices, int learnSampleCount, TVector<TVector<double>>* treeValues) {
    TVector<int> weights((*treeValues)[0].ysize());
    for (int docIdx = 0; docIdx < learnSampleCount; ++docIdx) {
        ++weights[indices[docIdx]];
    }

    double avrg = 0;
    for (int i = 0; i < weights.ysize(); ++i) {
        avrg += weights[i] * (*treeValues)[0][i];
    }

    int sumWeight = 0;
    for (int w : weights) {
        sumWeight += w;
    }
    avrg /= sumWeight;

    for (auto& value : (*treeValues)[0]) {
        value -= avrg;
    }
}
}

template <typename TError>
void UpdateLearningFold(
    const TDataset& learnData,
    const TDataset* testData,
    const TError& error,
    const TSplitTree& bestSplitTree,
    ui64 randomSeed,
    TFold* fold,
    TLearnContext* ctx
) {
    TVector<TVector<TVector<double>>> approxDelta;

    CalcApproxForLeafStruct(
        learnData,
        testData,
        error,
        *fold,
        bestSplitTree,
        randomSeed,
        ctx,
        &approxDelta
    );

    UpdateBodyTailApprox<TError::StoreExpApprox>(approxDelta, ctx->Params.BoostingOptions->LearningRate, &ctx->LocalExecutor, fold);
}

/// Change leave values in order to make tree monotonic on monotonicFeatures.
/// It's assumed that all monotonic splits are at the bottom levels of the tree.
template <typename TError>
void MonotonizeLeaveValues(TVector<TVector<double>>* leafValues,
                           const TSplitTree& tree,
                           const TTreeStats & treeStats,
                           TLearnContext* ctx,
                           const TVector<EMonotonicity> & monotonicFeatures)
{
    int numMonotonicSplits = 0;
    const auto & splits = tree.Splits;
    int numSplits = splits.ysize();
    while (numMonotonicSplits < numSplits) {
        const auto & split = splits[numSplits - numMonotonicSplits - 1];
        if (0 <= split.FeatureIdx && split.FeatureIdx < monotonicFeatures.ysize()
            && monotonicFeatures[split.FeatureIdx] != EMonotonicity::None)
            ++numMonotonicSplits;
        else
            break;
    }

    if (numMonotonicSplits == 0)
        return;

    TVector<EMonotonicity> monotonicity;
    monotonicity.reserve(numMonotonicSplits);
    for (int i = numSplits - numMonotonicSplits; i < numSplits; ++i)
        monotonicity.push_back(monotonicFeatures[splits[i].FeatureIdx]);

    auto isSplitViolatesMonotonicity = [](const double* leftLeaves, const double* rightLeaves,
                                          const double* leftWeights, const double* rightWeights,
                                          int numLeaves, int monDirection)
    {
        double leftExtremum = 0;
        double rightExtremum = 0;
        bool hasLeft = false;
        bool hasRight = false;
        for (int i = 0; i < numLeaves; ++i)
        {
            if (leftWeights[i] != 0 && (!hasLeft || leftExtremum * monDirection < leftLeaves[i] * monDirection)) {
                leftExtremum = leftLeaves[i];
                hasLeft = true;
            }
            if (rightWeights[i] != 0 && (!hasRight || rightExtremum * monDirection > rightLeaves[i] * monDirection)) {
                rightExtremum = rightLeaves[i];
                hasRight = true;
            }
        }

        return hasLeft && hasRight && leftExtremum * monDirection > rightExtremum * monDirection;
    };

    auto monotonizeSplit = [&](double* leftLeaves, double* rightLeaves,
                               const double* leftWeights, const double* rightWeights,
                               int numLeaves, int monDirection) -> void {

        if (!isSplitViolatesMonotonicity(leftLeaves, rightLeaves, leftWeights, rightWeights, numLeaves, monDirection))
            return;

        struct TLeaveStat {
            double Value;
            double Weight;
        };

        TVector<TLeaveStat> leftStats;
        TVector<TLeaveStat> rightStats;

        for (size_t i = 0; i < numLeaves; ++i) {
            if (leftWeights[i] != 0)
                leftStats[i].emplace_back(leftLeaves[i] * monDirection, leftWeights[i]);
            if (rightWeights[i] != 0)
                rightStats[i].emplace_back(rightLeaves[i] * monDirection, rightWeights[i]);
        }

        SortBy(leftStats, [monDirection](const TLeaveStat & stat) { return stat.Value; });
        SortBy(rightStats, [monDirection](const TLeaveStat & stat) { return stat.Value; });

        std::cerr << "Left values: ";
        for (auto & st : leftStats)
            std::cerr << '(' << st.Value << ", " << st.Weight << ") ";
        std::cerr << "\nRight values: ";
        for (auto & st : rightStats)
            std::cerr << '(' << st.Value << ", " << st.Weight << ") ";
        std::cerr << std::endl;

        /// Find optimal threshold:
        /// \sum{(left[i].value - threshold)^2 * left[i].weight * I[left[i].value > threshold]} +
        /// \sum{(threshold - right[i].value)^2 * right[i].weight * I[threshold > right[i].value]} -> min

        double threshold = std::min(leftStats[0].Value, rightStats[0].Value);

        struct TLossStat {
            double TotalWeight = 0;
            double L1 = 0;
            double L2 = 0;

            double getScore(double delta) const {
                return L2 + delta * delta * TotalWeight + 2.0 * delta * L1;
            }

            void AddShift(double delta) {
                L2 = getScore(delta);
                L1 += delta * TotalWeight;
            }
        };

        TLossStat leftLoss;
        TLossStat rightLoss;

        for (TLeaveStat & stat : leftStats) {
            double delta = (stat.Value - threshold);
            leftLoss.TotalWeight += stat.Weight;
            leftLoss.L1 += stat.Weight * delta;
            leftLoss.L2 += stat.Weight * delta * delta;
        }

        int leftIdx = 0;
        int rightIdx = 0;
        double bestThreshold = threshold;
        double bestScore = leftLoss.L2;

        while (leftIdx < numLeaves || rightIdx < numLeaves) {

            bool nextFromLeft = rightIdx >= numLeaves
                                || (leftIdx < numLeaves && leftStats[leftIdx].Value < rightStats[rightIdx].Value);
            double nextThreshold = nextFromLeft ? leftStats[leftIdx].Value
                                                : rightStats[rightIdx].Value;

            double delta = nextThreshold - threshold;
            double weight = leftLoss.TotalWeight + rightLoss.TotalWeight;
            double alpha = (leftLoss.L1 - rightLoss.L1) / std::max<double>(1, weight);
            alpha = std::max<double>(0, std::min<double>(1, alpha));
            double score = leftLoss.getScore(-alpha * delta) + rightLoss.getScore(alpha * delta);

            std::cerr << '\n' << leftIdx << '\t' << rightIdx << '\t' << alpha
                      << '\t' << leftLoss.getScore(-alpha * delta) << '\t' << rightLoss.getScore(alpha * delta)
                      << '\t' << score
                      << '\t' << leftLoss.TotalWeight << '\t' << rightLoss.TotalWeight
                      << '\t' << leftLoss.L1 << '\t' << rightLoss.L1 << std::endl;

            if (score < bestScore) {
                bestScore = score;
                bestThreshold = threshold + alpha * delta;
            }

            leftLoss.AddShift(-delta);
            rightLoss.AddShift(delta);

            if (nextFromLeft) {
                leftLoss.TotalWeight -= leftStats[leftIdx].Weight;
                ++leftIdx;
            } else {
                rightLoss.TotalWeight += rightStats[rightIdx].Weight;
                ++rightIdx;
            }

            threshold = nextThreshold;
        }

        std::cerr << "Selected th: " << bestThreshold << '\n' << std::endl;

        for (size_t i = 0; i < numLeaves; ++i) {
            if (leftLeaves[i] * monDirection > bestThreshold * monDirection)
                leftLeaves[i] = bestThreshold;
            if (rightLeaves[i] * monDirection < bestThreshold * monDirection)
                rightLeaves[i] = bestThreshold;
        }
    };

    const auto approxDimension = ctx->LearnProgress.AveragingFold.GetApproxDimension();
    const auto leafCount = tree.GetLeafCount();

    for (int dim = 0; dim < approxDimension; ++dim)
    {
        for (int depth = 0; depth < monotonicity.size(); ++depth)
        {
            int direction = static_cast<int>(monotonicity[depth]);
            int numLeaves = 1 << (monotonicity.ysize() - 1 - depth);
            for (int leaf = 0; leaf < leafCount; leaf += 2 * numLeaves)
            {
                double* leftLeaves = &(*leafValues)[dim][leaf];
                double* rightLeaves = &(*leafValues)[dim][leaf + numLeaves];
                const double* leftWeights = &treeStats.LeafWeightsSum[leaf];
                const double* rightWeights = &treeStats.LeafWeightsSum[leaf + numLeaves];

                monotonizeSplit(leftLeaves, rightLeaves, leftWeights, rightWeights, numLeaves, direction);
            }
        }
    }
}

template <typename TError>
void UpdateLeavesApproxes(
        const TDataset& learnData,
        const TDataset* testData,
        const TError& error,
        const TSplitTree& bestSplitTree,
        TLearnContext* ctx,
        TVector<TVector<double>>* treeValues,
        const TVector<TIndexType> & indices)
{
    TProfileInfo& profile = ctx->Profile;
    const int approxDimension = ctx->LearnProgress.AvrgApprox.ysize();
    const double learningRate = ctx->Params.BoostingOptions->LearningRate;
    const auto sampleCount = learnData.GetSampleCount() + (testData ? testData->GetSampleCount() : 0);

    TVector<TVector<double>> expTreeValues;
    expTreeValues.yresize(approxDimension);
    for (int dim = 0; dim < approxDimension; ++dim) {
        expTreeValues[dim] = (*treeValues)[dim];
        ExpApproxIf(TError::StoreExpApprox, &expTreeValues[dim]);
    }

    profile.AddOperation("CalcApprox result leafs");
    CheckInterrupted(); // check after long-lasting operation

    Y_ASSERT(ctx->LearnProgress.AveragingFold.BodyTailArr.ysize() == 1);
    TFold::TBodyTail& bt = ctx->LearnProgress.AveragingFold.BodyTailArr[0];

    const int tailFinish = bt.TailFinish;
    const int learnSampleCount = learnData.GetSampleCount();
    const size_t* learnPermutationData = ctx->LearnProgress.AveragingFold.LearnPermutation.data();
    const TIndexType* indicesData = indices.data();
    for (int dim = 0; dim < approxDimension; ++dim) {
        const double* expTreeValuesData = expTreeValues[dim].data();
        const double* treeValuesData = (*treeValues)[dim].data();
        double* approxData = bt.Approx[dim].data();
        double* avrgApproxData = ctx->LearnProgress.AvrgApprox[dim].data();
        double* testApproxData = ctx->LearnProgress.TestApprox[dim].data();
        ctx->LocalExecutor.ExecRange(
                [=](int docIdx) {
                    const int permutedDocIdx = docIdx < learnSampleCount ? learnPermutationData[docIdx] : docIdx;
                    if (docIdx < tailFinish) {
                        Y_VERIFY(docIdx < learnSampleCount);
                        approxData[docIdx] = UpdateApprox<TError::StoreExpApprox>(approxData[docIdx], expTreeValuesData[indicesData[docIdx]]);
                    }
                    if (docIdx < learnSampleCount) {
                        avrgApproxData[permutedDocIdx] += treeValuesData[indicesData[docIdx]];
                    } else {
                        testApproxData[docIdx - learnSampleCount] += treeValuesData[indicesData[docIdx]];
                    }
                },
                NPar::TLocalExecutor::TExecRangeParams(0, sampleCount).SetBlockSize(1000),
                NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }
}

template <typename TError>
void UpdateTreeLeaves(const TDataset& learnData,
                      const TDataset* testData,
                      const TError& error,
                      const TSplitTree& splitTree,
                      TLearnContext* ctx,
                      TVector<TVector<double>>* treeValues,
                      const TTreeStats & treeStats,
                      const TVector<EMonotonicity> & monotonicFeatures)
{
    TVector<TIndexType> indices;

    TVector<TVector<double>> newValues;

    CalcLeafValues(
            learnData,
            testData,
            error,
            ctx->LearnProgress.AveragingFold,
            splitTree,
            ctx,
            &newValues,
            &indices
    );

    MonotonizeLeaveValues<TError>(&newValues, splitTree, treeStats, ctx, monotonicFeatures);

    const double learningRate = ctx->Params.BoostingOptions->LearningRate;

    for (int dim = 0; dim < treeValues->ysize(); ++dim) {
        auto & treeDim = (*treeValues)[dim];
        auto & newTreeDim = newValues[dim];
        for (size_t leave = 0; leave < treeDim.ysize(); ++leave) {
            auto leaveVal = newTreeDim[leave] * learningRate;
            newTreeDim[leave] = leaveVal - treeDim[leave];
            treeDim[leave] = leaveVal;
        }
    }

    UpdateLeavesApproxes(learnData, testData, error, splitTree, ctx, &newValues, indices);
}

template <typename TError>
void UpdateAveragingFold(
    const TDataset& learnData,
    const TDataset* testData,
    const TError& error,
    const TSplitTree& bestSplitTree,
    TLearnContext* ctx,
    TVector<TVector<double>>* treeValues,
    const TVector<EMonotonicity> & monotonicFeatures
) {
    TProfileInfo& profile = ctx->Profile;
    TVector<TIndexType> indices;

    CalcLeafValues(
        learnData,
        testData,
        error,
        ctx->LearnProgress.AveragingFold,
        bestSplitTree,
        ctx,
        treeValues,
        &indices
    );

    const int approxDimension = ctx->LearnProgress.AvrgApprox.ysize();
    const double learningRate = ctx->Params.BoostingOptions->LearningRate;
    const auto sampleCount = learnData.GetSampleCount() + (testData ? testData->GetSampleCount() : 0);

    for (int dim = 0; dim < approxDimension; ++dim) {
        for (auto & leafVal : (*treeValues)[dim]) {
            leafVal *= learningRate;
        }
    }

    auto& currentTreeStats = ctx->LearnProgress.TreeStats.emplace_back();
    currentTreeStats.LeafWeightsSum.resize((*treeValues)[0].size());
    for (auto docId = 0; docId < learnData.GetSampleCount(); ++docId) {
        currentTreeStats.LeafWeightsSum[indices[ctx->LearnProgress.AveragingFold.LearnPermutation[docId]]] += learnData.Weights[docId];
    }
    // TODO(nikitxskv): if this will be a bottleneck, we can use precalculated counts.
    if (IsPairwiseError(ctx->Params.LossFunctionDescription->GetLossFunction())) {
        NormalizeLeafValues(indices, learnData.GetSampleCount(), treeValues);
    }

    MonotonizeLeaveValues<TError>(treeValues, bestSplitTree, currentTreeStats, ctx, monotonicFeatures);

    UpdateLeavesApproxes(learnData, testData, error, bestSplitTree, ctx, treeValues, indices);
}

/// Move monotonic features to the end of list. Return the number of monotonic features.
template <typename TError>
void SiftDownMonotonicSplits(TSplitTree* splitTree, const TVector<EMonotonicity> & monotonicFeatures)
{
    std::cerr << "Mon: ";
    for (auto f : monotonicFeatures)
        std::cerr << static_cast<int>(f) << ' ';
    std::cerr << std::endl << "Indexes: ";
    for (auto & spl : splitTree->Splits)
        std::cerr << spl.FeatureIdx << ' ';
    std::cerr << std::endl;

    auto splits = splitTree->Splits;
    auto numSplits = splits.ysize();

    auto isMonotonicFeature = [&](int splitIdx) {
        int featureIdx = splits[splitIdx].FeatureIdx;
        return 0 <= featureIdx && featureIdx < monotonicFeatures.ysize() && monotonicFeatures[featureIdx] != EMonotonicity::None;
    };

    int firstMonotonic = 0;
    int lastNotMonotonic = splits.ysize() - 1;
    do {
        while (firstMonotonic < numSplits && !isMonotonicFeature(firstMonotonic))
            ++firstMonotonic;
        while (0 <= lastNotMonotonic && isMonotonicFeature(lastNotMonotonic))
            --lastNotMonotonic;

        if (firstMonotonic < lastNotMonotonic) {
            std::swap(splits[firstMonotonic], splits[lastNotMonotonic]);
            ++firstMonotonic;
            --lastNotMonotonic;
        }
    } while (firstMonotonic < lastNotMonotonic);
}

template <typename TError>
void TrainOneIter(const TDataset& learnData, const TDataset* testData, TLearnContext* ctx) {
    TError error = BuildError<TError>(ctx->Params, ctx->ObjectiveDescriptor);
    TProfileInfo& profile = ctx->Profile;

    const TVector<int> splitCounts = CountSplits(ctx->LearnProgress.FloatFeatures);

    const int foldCount = ctx->LearnProgress.Folds.ysize();
    const int currentIteration = ctx->LearnProgress.TreeStruct.ysize();
    const double modelLength = currentIteration * ctx->Params.BoostingOptions->LearningRate;

    CheckInterrupted(); // check after long-lasting operation

    TSplitTree bestSplitTree;
    {
        TFold* takenFold = &ctx->LearnProgress.Folds[ctx->Rand.GenRand() % foldCount];
        const TVector<ui64> randomSeeds = GenRandUI64Vector(takenFold->BodyTailArr.ysize(), ctx->Rand.GenRand());
        if (ctx->Params.SystemOptions->IsSingleHost()) {
            ctx->LocalExecutor.ExecRange([&](int bodyTailId) {
                CalcWeightedDerivatives(error, bodyTailId, ctx->Params, randomSeeds[bodyTailId], takenFold, &ctx->LocalExecutor);
            }, 0, takenFold->BodyTailArr.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
        } else {
            Y_ASSERT(takenFold->BodyTailArr.ysize() == 1);
            MapSetDerivatives<TError>(ctx);
        }
        profile.AddOperation("Calc derivatives");

        GreedyTensorSearch(
            learnData,
            testData,
            splitCounts,
            modelLength,
            profile,
            takenFold,
            ctx,
            &bestSplitTree
        );
    }
    CheckInterrupted(); // check after long-lasting operation

    const auto & monotonicFeatures = ctx->Params.DataProcessingOptions->MonotonicFeatures.Get();
    if (!monotonicFeatures.empty())
        SiftDownMonotonicSplits<TError>(&bestSplitTree, monotonicFeatures);

    {
        TVector<TFold*> trainFolds;
        for (int foldId = 0; foldId < foldCount; ++foldId) {
            trainFolds.push_back(&ctx->LearnProgress.Folds[foldId]);
        }

        TrimOnlineCTRcache(trainFolds);
        TrimOnlineCTRcache({ &ctx->LearnProgress.AveragingFold });
        {
            TVector<TFold*> allFolds = trainFolds;
            allFolds.push_back(&ctx->LearnProgress.AveragingFold);

            struct TLocalJobData {
                const TDataset* LearnData;
                const TDataset* TestData;
                TProjection Projection;
                TFold* Fold;
                TOnlineCTR* Ctr;
                void DoTask(TLearnContext* ctx) {
                    ComputeOnlineCTRs(*LearnData, TestData, *Fold, Projection, ctx, Ctr);
                }
            };

            TVector<TLocalJobData> parallelJobsData;
            THashSet<TProjection> seenProjections;
            for (const auto& split : bestSplitTree.Splits) {
                if (split.Type != ESplitType::OnlineCtr) {
                    continue;
                }

                const auto& proj = split.Ctr.Projection;
                if (seenProjections.has(proj)) {
                    continue;
                }
                for (auto* foldPtr : allFolds) {
                    if (!foldPtr->GetCtrs(proj).has(proj) || foldPtr->GetCtr(proj).Feature.empty()) {
                        parallelJobsData.emplace_back(TLocalJobData{ &learnData, testData, proj, foldPtr, &foldPtr->GetCtrRef(proj) });
                    }
                }
                seenProjections.insert(proj);
            }

            ctx->LocalExecutor.ExecRange([&](int taskId){
                parallelJobsData[taskId].DoTask(ctx);
            }, 0, parallelJobsData.size(), NPar::TLocalExecutor::WAIT_COMPLETE);

        }
        profile.AddOperation("ComputeOnlineCTRs for tree struct (train folds and test fold)");
        CheckInterrupted(); // check after long-lasting operation

        if (ctx->Params.SystemOptions->IsSingleHost()) {
            const TVector<ui64> randomSeeds = GenRandUI64Vector(foldCount, ctx->Rand.GenRand());
            ctx->LocalExecutor.ExecRange([&](int foldId) {
                UpdateLearningFold(learnData, testData, error, bestSplitTree, randomSeeds[foldId], trainFolds[foldId], ctx);
            }, 0, foldCount, NPar::TLocalExecutor::WAIT_COMPLETE);
        } else {
            MapSetApproxes<TError>(bestSplitTree, ctx);
        }

        profile.AddOperation("CalcApprox tree struct and update tree structure approx");
        CheckInterrupted(); // check after long-lasting operation

        TVector<TVector<double>> treeValues; // [dim][leafId]
        UpdateAveragingFold(learnData, testData, error, bestSplitTree, ctx, &treeValues, monotonicFeatures);

        ctx->LearnProgress.LeafValues.push_back(treeValues);
        ctx->LearnProgress.TreeStruct.push_back(bestSplitTree);

        profile.AddOperation("Update final approxes");
        CheckInterrupted(); // check after long-lasting operation

        auto numAddedTrees = ctx->LearnProgress.TreeStruct.ysize();
        if (!monotonicFeatures.empty() && numAddedTrees > 1) {
            auto randTreeIndex = ctx->Rand.GenRand() % numAddedTrees;
            const auto & randTree = ctx->LearnProgress.TreeStruct[randTreeIndex];
            auto & randTreeValues =  ctx->LearnProgress.LeafValues[randTreeIndex];
            const auto & randTreeStats = ctx->LearnProgress.TreeStats[randTreeIndex];
            UpdateTreeLeaves(learnData, testData, error, randTree, ctx, &randTreeValues, randTreeStats, monotonicFeatures);

            profile.AddOperation("Update random tree leaves");
            CheckInterrupted(); // check after long-lasting operation
        }

        std::cerr << "Result Leaves:" << std::endl;
        for (auto & vals : treeValues)
        {
            std::cerr << "dim: ";
            for (auto val : vals)
                std::cerr << val << ' ';
            std::cerr << std::endl;
        }
    }
}
