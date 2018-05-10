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


template <typename TError>
TVector<EMonotonicity> GetTreeMonotonicFeatures(const TSplitTree& tree, const TVector<EMonotonicity> & monotonicFeatures) {
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

    TVector<EMonotonicity> monotonicity;
    monotonicity.reserve(numMonotonicSplits);
    for (int i = numSplits - numMonotonicSplits; i < numSplits; ++i)
        monotonicity.push_back(monotonicFeatures[splits[i].FeatureIdx]);

    return monotonicity;
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
//    int numMonotonicSplits = 0;
//    const auto & splits = tree.Splits;
//    int numSplits = splits.ysize();
//    while (numMonotonicSplits < numSplits) {
//        const auto & split = splits[numSplits - numMonotonicSplits - 1];
//        if (0 <= split.FeatureIdx && split.FeatureIdx < monotonicFeatures.ysize()
//            && monotonicFeatures[split.FeatureIdx] != EMonotonicity::None)
//            ++numMonotonicSplits;
//        else
//            break;
//    }
//
//    if (numMonotonicSplits == 0)
//        return;
//
//    TVector<EMonotonicity> monotonicity;
//    monotonicity.reserve(numMonotonicSplits);
//    for (int i = numSplits - numMonotonicSplits; i < numSplits; ++i)
//        monotonicity.push_back(monotonicFeatures[splits[i].FeatureIdx]);

    if (monotonicFeatures.empty())
        return;

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
                leftStats.push_back({leftLeaves[i] * monDirection, leftWeights[i]});
            if (rightWeights[i] != 0)
                rightStats.push_back({rightLeaves[i] * monDirection, rightWeights[i]});
        }

        int numLeft = leftStats.ysize();
        int numRight = rightStats.ysize();

        SortBy(leftStats, [monDirection](const TLeaveStat & stat) { return stat.Value; });
        SortBy(rightStats, [monDirection](const TLeaveStat & stat) { return stat.Value; });

//        std::cerr << "Left values: ";
//        for (auto & st : leftStats)
//            std::cerr << '(' << st.Value << ", " << st.Weight << ") ";
//        std::cerr << "\nRight values: ";
//        for (auto & st : rightStats)
//            std::cerr << '(' << st.Value << ", " << st.Weight << ") ";
//        std::cerr << std::endl;

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

        while (leftIdx < numLeft || rightIdx < numRight) {

            bool nextFromLeft = rightIdx >= numRight
                                || (leftIdx < numLeft && leftStats[leftIdx].Value < rightStats[rightIdx].Value);
            double nextThreshold = nextFromLeft ? leftStats[leftIdx].Value
                                                : rightStats[rightIdx].Value;

            double delta = nextThreshold - threshold;
            double weight = leftLoss.TotalWeight + rightLoss.TotalWeight;
            double alpha = (leftLoss.L1 - rightLoss.L1) / std::max<double>(1, weight);
            alpha = std::max<double>(0, std::min<double>(1, alpha));
            double score = leftLoss.getScore(-alpha * delta) + rightLoss.getScore(alpha * delta);

//            std::cerr << '\n' << leftIdx << '\t' << rightIdx << '\t' << alpha
//                      << '\t' << leftLoss.getScore(-alpha * delta) << '\t' << rightLoss.getScore(alpha * delta)
//                      << '\t' << score
//                      << '\t' << leftLoss.TotalWeight << '\t' << rightLoss.TotalWeight
//                      << '\t' << leftLoss.L1 << '\t' << rightLoss.L1 << std::endl;

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

//        std::cerr << "Selected th: " << bestThreshold << '\n' << std::endl;

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
        for (int depth = 0; depth < monotonicFeatures.size(); ++depth)
        {
            int direction = static_cast<int>(monotonicFeatures[depth]);
            int numLeaves = 1 << (monotonicFeatures.ysize() - 1 - depth);
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

template<bool StoreExpApprox, bool RollbackTree = false>
void UpdateLeafApproxes(
        const TDataset & learnData,
        const TDataset * testData,
        const TSplitTree & bestSplitTree,
        TLearnContext * ctx,
        const TVector<TVector<double>> & treeValues,
        const TVector<TIndexType> & indices)
{
    TProfileInfo& profile = ctx->Profile;
    const int approxDimension = ctx->LearnProgress.AvrgApprox.ysize();
//     const double learningRate = ctx->Params.BoostingOptions->LearningRate;
    const auto sampleCount = learnData.GetSampleCount() + (testData ? testData->GetSampleCount() : 0);

    constexpr double RollbackSign = RollbackTree ? -1 : 1;

    TVector<TVector<double>> expTreeValues;
    expTreeValues.yresize(approxDimension);
    for (int dim = 0; dim < approxDimension; ++dim) {
        expTreeValues[dim] = treeValues[dim];
        if (RollbackTree) {
            for (auto & val : expTreeValues[dim])
                val = -val;
        }
        ExpApproxIf(StoreExpApprox, &expTreeValues[dim]);
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
        const double* treeValuesData = treeValues[dim].data();
        double* approxData = bt.Approx[dim].data();
        double* avrgApproxData = ctx->LearnProgress.AvrgApprox[dim].data();
        double* testApproxData = ctx->LearnProgress.TestApprox[dim].data();
        ctx->LocalExecutor.ExecRange(
                [=](int docIdx) {
                    const int permutedDocIdx = docIdx < learnSampleCount ? learnPermutationData[docIdx] : docIdx;
                    if (docIdx < tailFinish) {
                        Y_VERIFY(docIdx < learnSampleCount);
                        approxData[docIdx] = UpdateApprox<StoreExpApprox>(approxData[docIdx], expTreeValuesData[indicesData[docIdx]]);
                    }
                    if (docIdx < learnSampleCount) {
                        avrgApproxData[permutedDocIdx] += treeValuesData[indicesData[docIdx]] * RollbackSign;
                    } else {
                        testApproxData[docIdx - learnSampleCount] += treeValuesData[indicesData[docIdx]] * RollbackSign;
                    }
                },
                NPar::TLocalExecutor::TExecRangeParams(0, sampleCount).SetBlockSize(1000),
                NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }
}

template<bool StoreExpApprox>
void SmoothApproxes(
        const TDataset & learnData,
        const TDataset * testData,
        TLearnContext * ctx)
{
    TProfileInfo& profile = ctx->Profile;
    const int approxDimension = ctx->LearnProgress.AvrgApprox.ysize();
    const auto sampleCount = learnData.GetSampleCount() + (testData ? testData->GetSampleCount() : 0);

    TVector<double> totalMean(approxDimension, 0);
    TVector<double> avgVar(approxDimension, 0);
    if (!ctx->LearnProgress.TreeStats.empty()) {
        for (const auto & stat : ctx->LearnProgress.TreeStats) {
            if (!stat.LeafMean.empty()) {
                for (int dim = 0; dim < approxDimension; ++dim)
                    totalMean[dim] += stat.LeafMean[dim];
            }
            if (!stat.LeafVar.empty()) {
                for (int dim = 0; dim < approxDimension; ++dim)
                    avgVar[dim] += stat.LeafVar[dim];
            }
        }
        for (auto & var : avgVar)
            var = sqrt(var / ctx->LearnProgress.TreeStats.size());
    }

    Y_ASSERT(ctx->LearnProgress.AveragingFold.BodyTailArr.ysize() == 1);
    TFold::TBodyTail& bt = ctx->LearnProgress.AveragingFold.BodyTailArr[0];

    const int tailFinish = bt.TailFinish;
    const int learnSampleCount = learnData.GetSampleCount();
    const size_t* learnPermutationData = ctx->LearnProgress.AveragingFold.LearnPermutation.data();
    for (int dim = 0; dim < approxDimension; ++dim) {

        double smooth = ctx->Params.BoostingOptions->LearningRate.Get() / 10.0;
        double weight = /*avgVar[dim] */ (1.0 - smooth);
        double smoothedMean = totalMean[dim] * smooth;
        double expSmoothedMean = fast_exp(smoothedMean);
        double* approxData = bt.Approx[dim].data();
        double* avrgApproxData = ctx->LearnProgress.AvrgApprox[dim].data();
        double* testApproxData = ctx->LearnProgress.TestApprox[dim].data();
        ctx->LocalExecutor.ExecRange(
                [=](int docIdx) {
                    const int permutedDocIdx = docIdx < learnSampleCount ? learnPermutationData[docIdx] : docIdx;
                    if (docIdx < tailFinish) {
                        Y_VERIFY(docIdx < learnSampleCount);
                        approxData[docIdx] = StoreExpApprox
                                             ? expSmoothedMean * fast_exp(weight * log(approxData[docIdx]))
                                             : approxData[docIdx] * weight + smoothedMean;
                    }
                    if (docIdx < learnSampleCount) {
                        avrgApproxData[permutedDocIdx] = avrgApproxData[permutedDocIdx] * weight + smoothedMean;
                    } else {
                        testApproxData[docIdx - learnSampleCount] *= weight;
                        testApproxData[docIdx - learnSampleCount] += smoothedMean;
                    }
                },
                NPar::TLocalExecutor::TExecRangeParams(0, sampleCount).SetBlockSize(1000),
                NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }
}

template <typename Error>
TVector<double> EvalMetricPerLeaf(const TDataset & learnData,
                                  const TSplitTree & tree,
                                  TLearnContext * ctx,
                                  const THolder<IMetric> & metric,
                                  const TVector<TVector<double>> * treeValues,
                                  int numLeafs,
                                  const TVector<TIndexType> & indices)
{
    TVector<double> metricPerLeaf(numLeafs, 0);
    TVector<TVector<int>> leafsIndices(numLeafs);

    const auto& avrgApprox = ctx->LearnProgress.AvrgApprox;
    const float* learnTarget = learnData.Target.data(); // ctx->LearnProgress.AveragingFold.LearnTarget.data();
    const float* learnWeight = learnData.Weights.data(); // ctx->LearnProgress.AveragingFold.LearnWeights.data();
    const auto& learnQueriesInfo = learnData.QueryInfo; //ctx->LearnProgress.AveragingFold.LearnQueriesInfo;
    //const size_t* learnPermutationData = ctx->LearnProgress.AveragingFold.LearnPermutation.data();

    const int approxDimension = avrgApprox.ysize();
    const auto learnSampleCount = learnData.GetSampleCount();

    for (int doc = 0; doc < indices.ysize(); ++doc)
        if (doc < learnSampleCount)
            leafsIndices[indices[doc]].push_back(doc);

    for (int leaf = 0; leaf < numLeafs; ++leaf) {
        const auto numDocs = leafsIndices[leaf].size();

//        std::cerr << "Ind leaf " << leaf;
//        for (auto val : leafsIndices[leaf])
//            std::cerr << ' ' << val;
//        std::cerr << std::endl;

        if (numDocs == 0)
            continue;

        TVector<TVector<double>> approx(approxDimension);
        TVector<float> target(numDocs);
        TVector<float> weight(numDocs);
        //TVector<TQueryInfo> queriesInfo(numDocs);

        for (int dim = 0; dim < approxDimension; ++dim)
            approx[dim].resize(numDocs);

        for (int doc = 0; doc < numDocs; ++doc)
        {
            const auto docIdx = leafsIndices[leaf][doc];
            //const auto permutedDocIdx = learnPermutationData[docIdx];
            target[doc] = learnTarget[doc]; //[permutedDocIdx];
            weight[doc] = learnWeight[doc]; //[permutedDocIdx];

            for (int dim = 0; dim < approxDimension; ++dim)
                approx[dim][doc] = avrgApprox[dim][doc]; //[permutedDocIdx];

//            if (!learnQueriesInfo.empty())
//                queriesInfo[doc] = learnQueriesInfo[permutedDocIdx];
        }

        if (treeValues) {
            for (auto & values : approx) {
                for (int dim = 0; dim < approxDimension; ++dim)
                    values[dim] += (*treeValues)[dim][leaf];
            }
        }

        metricPerLeaf[leaf] = EvalErrors(approx, target, weight, learnQueriesInfo, metric, &ctx->LocalExecutor, false);
    }

    return metricPerLeaf;
}

template <typename TError>
void SmoothTrees(const TDataset& learnData,
                 const TDataset* testData,
                 TLearnContext* ctx)
{
    int numTrees = ctx->LearnProgress.TreeStats.ysize();
    if (numTrees < 5)
        return;

    int approxDimension = ctx->LearnProgress.ApproxDimension;
    TVector<double> avgVar(approxDimension, 0);

    for (const auto & stat : ctx->LearnProgress.TreeStats) {
        if (!stat.LeafVar.empty()) {
            for (int dim = 0; dim < approxDimension; ++dim)
                avgVar[dim] += stat.LeafVar[dim];
        }
    }
    for (auto & var : avgVar)
        var = var / numTrees;

    int treeIdx = 0;
    double maxVar = 0;

    for (int tree = 0; tree < numTrees; ++tree) {
        const auto & stat = ctx->LearnProgress.TreeStats[tree];
        double var = 0;
        if (!stat.LeafVar.empty()) {
            for (int dim = 0; dim < approxDimension; ++dim) {
                if (avgVar[dim] != 0)
                    var = std::max(var, stat.LeafVar[dim] / avgVar[dim]);
            }
        }
        if (maxVar < var) {
            maxVar = var;
            treeIdx = tree;
        }
    }

    auto & tree = ctx->LearnProgress.TreeStruct[treeIdx];
    auto & stat = ctx->LearnProgress.TreeStats[treeIdx];
    auto & treeValues = ctx->LearnProgress.LeafValues[treeIdx];
    auto indices = BuildIndices(ctx->LearnProgress.AveragingFold, tree, learnData, testData, &ctx->LocalExecutor);
    UpdateLeafApproxes<TError::StoreExpApprox, true>(learnData, testData, tree, ctx, treeValues, indices);

    for (int dim = 0; dim < approxDimension; ++dim) {
        if (avgVar[dim] != 0) {
            auto & leafs = treeValues[dim];
            double mean = stat.LeafMean.empty() ? 0 : stat.LeafMean[dim];
            double var = stat.LeafVar.empty() ? 0 : stat.LeafVar[dim];
            double smooth = sqrt(avgVar[dim] / var);
            std::cerr << "tree: " << treeIdx << " var " << var << " avg var " << avgVar[dim] << " smooth " << smooth << std::endl;
            for (auto & leaf : leafs)
                leaf = (leaf - mean) * smooth + mean;
        }
    }

    stat.CalcStats(treeValues);
    UpdateLeafApproxes<TError::StoreExpApprox>(learnData, testData, tree, ctx, treeValues, indices);
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

    UpdateLeafApproxes<TError::StoreExpApprox>(learnData, testData, splitTree, ctx, newValues, indices);
}


template <typename TError>
void PruneTreeNodes(TVector<double> & prevLoss,
                    TVector<double> & currLoss,
                    const TVector<EMonotonicity> & monotonicFeatures,
                    TVector<TVector<double>> * leafValues,
                    const THolder<IMetric> & metric) {
    EMetricBestValue valueType;
    float bestValue;
    metric->GetBestValue(&valueType, &bestValue);

    auto isLossGotWorse = [&](double prev, double curr) {
        if (valueType == EMetricBestValue::Max)
            return curr < prev;
        if (valueType == EMetricBestValue::Min)
            return curr > prev;
        if (valueType == EMetricBestValue::FixedValue)
            return fabs(curr - bestValue) > fabs(prev - bestValue);
        return false;
    };

    int numDims = leafValues->ysize();
    int numLeafs = leafValues->at(0).ysize();

    auto setLeafsLoverBound = [&](int start, int count, double bound) {
        for (int dim = 0; dim < numDims; ++dim) {
            auto & dimValues = (*leafValues)[dim];

            for (int leave = start; leave < start + count; ++leave)
                dimValues[leave] = std::max(bound, dimValues[leave]);
        }
    };

    auto setLeafsUpperBound = [&](int start, int count, double bound) {
        for (int dim = 0; dim < numDims; ++dim) {
            auto & dimValues = (*leafValues)[dim];

            for (int leave = start; leave < start + count; ++leave)
                dimValues[leave] = std::min(bound, dimValues[leave]);
        }
    };

    std::function<bool(int, int, int)> prune;
    prune = [&](int start, int subtreeSize, int currMonotonicFeature) {
        if (subtreeSize == 1) {
            bool lossGotWorse = isLossGotWorse(prevLoss[start], currLoss[start]);
            if (lossGotWorse) {
                for (int dim = 0; dim < numDims; ++dim)
                    (*leafValues)[dim][start] = 0;
            }
            return lossGotWorse;
        }

        auto childSize = subtreeSize / 2;
        bool leftSubtreePruned = prune(start, childSize, currMonotonicFeature + 1);
        bool rightSubtreePruned = prune(start + childSize, childSize, currMonotonicFeature + 1);

        // std::cerr << "Cnt: " << subtreeSize << " lp " << leftSubtreePruned << " rp " << rightSubtreePruned << std::endl;

        if (leftSubtreePruned) {
            if (monotonicFeatures[currMonotonicFeature] == EMonotonicity::Ascending)
                setLeafsLoverBound(start + childSize, childSize, 0);
            else
                setLeafsUpperBound(start + childSize, childSize, 0);
        }
        if (rightSubtreePruned) {
            if (monotonicFeatures[currMonotonicFeature] == EMonotonicity::Ascending)
                setLeafsUpperBound(start, childSize, 0);
            else
                setLeafsLoverBound(start, childSize, 0);
        }

        return leftSubtreePruned || rightSubtreePruned;
    };

    int numLeafsInMostRestrictedSubtree = 1 << monotonicFeatures.ysize();
    for (int start = 0; start < numLeafs; start += numLeafsInMostRestrictedSubtree)
        prune(start, numLeafsInMostRestrictedSubtree, 0);
}

template <typename TError>
TVector<TVector<TVector<double>>> CalcLeafValuesAllLayers(
    TSplitTree tree,
    const TDataset& learnData,
    const TDataset* testData,
    const TError& error,
    TLearnContext* ctx
) {
    TVector<TVector<TVector<double>>> allLayersLeafValues(tree.Splits.size());
    while (!tree.Splits.empty()) {
        tree.Splits.erase(tree.Splits.begin());
        TVector<TIndexType> indices;
        CalcLeafValues(
                learnData,
                testData,
                error,
                ctx->LearnProgress.AveragingFold,
                tree,
                ctx,
                &allLayersLeafValues[tree.Splits.size()],
                &indices
        );
    }
    return allLayersLeafValues;
}

template <typename TError>
void MonotonizeAllLayers2(
        const TVector<EMonotonicity> & treeMonotonicFeatures,
        TVector<TVector<TVector<double>>>* layersValues,
        TVector<TVector<double>>* leafValues,
        const TSplitTree& tree,
        const TTreeStats& treeStats
) {
    int numDims = leafValues->ysize();
    int numSplits = tree.Splits.ysize();
    int numNotMonotonicSplits = numSplits - treeMonotonicFeatures.ysize();
    TVector<TVector<double>> layersWeights(numSplits);

    /// Calc weights for all nodes.
    for (int depth = numSplits; depth > 0; --depth) {
        TVector<double> & curLayerWeights = layersWeights[depth - 1];
        const auto& nextLayerWeights = depth == numSplits ? treeStats.LeafWeightsSum : layersWeights[depth];
        int numNodes = 1 << (depth - 1);
        curLayerWeights.resize(numNodes);
        for (int node = 0; node < numNodes; ++node)
            curLayerWeights[node] = nextLayerWeights[2 * node] + nextLayerWeights[2 * node + 1];
    }

    struct TMinMaxStats {
        double MinValue = -std::numeric_limits<double>::max();
        double MaxValue = std::numeric_limits<double>::max();
    };

    /// Monotonise values in all layers consistently.
    for (int dim = 0; dim < numDims; ++dim) {
        TVector<TMinMaxStats> curLayerMinMax(1);
        for (int depth = 0; depth <= numSplits; ++depth)
        {
            std::cerr << "L " << depth << " MinMax:";
            for (auto & mm : curLayerMinMax)
                std::cerr << " (" << mm.MinValue << ", " << mm.MaxValue << ")";
            std::cerr << std::endl;

            int numNodes = 1 << depth;
            TVector<double> & curLevelValues = depth == numSplits ? (*leafValues)[dim]
                                                                  : (*layersValues)[depth][dim];
            for (int node = 0; node < numNodes; ++node) {
                const auto & minMax = curLayerMinMax[node];
                curLevelValues[node] = std::max(minMax.MinValue, std::min(minMax.MaxValue, curLevelValues[node]));
            }

            if (depth != numSplits && depth >= numNotMonotonicSplits) {
                const auto mon = treeMonotonicFeatures[depth - numNotMonotonicSplits];
                TVector<TMinMaxStats> nextLayerMinMax(2 * numNodes);
                TVector<double> & nextLevelValues = depth + 1 == numSplits ? (*leafValues)[dim]
                                                                           : (*layersValues)[depth + 1][dim];
                for (int node = 0; node < numNodes; ++node) {
                    const auto & weights = depth + 1 == numSplits ? treeStats.LeafWeightsSum : layersWeights[depth + 1];
                    double leftWeights = weights[2 * node];
                    double rightWeights = weights[2 * node + 1];
                    double leftVal = nextLevelValues[2 * node];
                    double rightVal = nextLevelValues[2 * node + 1];
                    if (leftWeights + rightWeights > 0) {
                        double med = (leftWeights * leftVal + rightWeights * rightVal) / (leftWeights + rightWeights);
                        const auto & minMax = curLayerMinMax[node];
                        med = std::max(minMax.MinValue, std::min(minMax.MaxValue, med));
                        auto& leftMinMax = nextLayerMinMax[2 * node];
                        auto& rightMinMax = nextLayerMinMax[2 * node + 1];
                        leftMinMax = rightMinMax = curLayerMinMax[node];
                        if (mon == EMonotonicity::Ascending) {
                            leftMinMax.MaxValue = std::min(leftMinMax.MaxValue, med);
                            rightMinMax.MinValue = std::max(rightMinMax.MinValue, med);
                        } else {
                            leftMinMax.MinValue = std::max(leftMinMax.MinValue, med);
                            rightMinMax.MaxValue = std::min(rightMinMax.MaxValue, med);
                        }
                    }
                }
                curLayerMinMax.swap(nextLayerMinMax);
            }
        }
    }
}

template <typename TError>
void MonotonizeAllLayers(
    const TVector<EMonotonicity> & treeMonotonicFeatures,
    TVector<TVector<TVector<double>>> * layersValues,
    const TVector<TVector<double>> & leafValues,
    const TSplitTree & tree,
    const TTreeStats & treeStats
) {
    if (tree.Splits.empty())
        return;

    struct TMinMaxStats {
        double MinValue = -std::numeric_limits<double>::max();
        double MaxValue = std::numeric_limits<double>::max();

        void update(const TMinMaxStats & stats) {
            MinValue = std::min(MinValue, stats.MinValue);
            MaxValue = std::max(MaxValue, stats.MaxValue);
        }
    };

    int numSplits = tree.Splits.ysize();
    int numNotMonotonicSplits = numSplits - treeMonotonicFeatures.ysize();

    std::function<void(TMinMaxStats, int, int, int, const TVector<TVector<TMinMaxStats>> &)> monotonize;

    monotonize = [&](TMinMaxStats stats, int leaf, int depth, int dim, const TVector<TVector<TMinMaxStats>> & minMax) {
        if (depth >= numSplits)
            return;


        auto & val = (*layersValues)[depth][dim][leaf];
//        std::cerr << "Depth " << depth << " mns " << numNotMonotonicSplits << " leaf " << leaf << " val " << val << " stats (" << stats.MinValue << ", " << stats.MaxValue << ") ";
        val = std::min(val, stats.MaxValue);
        val = std::max(val, stats.MinValue);
//        std::cerr << " res " << val << std::endl;


        if (depth + 1 >= numSplits)
            return;

        if (depth < numNotMonotonicSplits) {
            monotonize(stats, 2 * leaf, depth + 1, dim, minMax);
            monotonize(stats, 2 * leaf + 1, depth + 1, dim, minMax);
        } else {
            const TMinMaxStats & leftMinMax = minMax[depth + 1][2 * leaf];
            const TMinMaxStats & rightMinMax = minMax[depth + 1][2 * leaf + 1];
            const auto mon = treeMonotonicFeatures[depth - numNotMonotonicSplits];
            if (mon == EMonotonicity::Ascending) {
                auto med = 0.5 * (leftMinMax.MaxValue + rightMinMax.MinValue);
                auto childStats = stats;
                childStats.MaxValue = std::min(childStats.MaxValue, med);
                std::cerr << "LS: (" << childStats.MinValue << ", " << childStats.MaxValue << ") ";
                monotonize(childStats, 2 * leaf, depth + 1, dim, minMax);
                childStats = stats;
                childStats.MinValue = std::max(childStats.MinValue, med);
                std::cerr << "RS: (" << childStats.MinValue << ", " << childStats.MaxValue << ") " << std::endl;
                monotonize(childStats, 2 * leaf + 1, depth + 1, dim, minMax);
            } else {
                auto med = 0.5 * (leftMinMax.MinValue + rightMinMax.MaxValue);
                auto childStats = stats;
                childStats.MinValue = std::max(childStats.MinValue, med);
                monotonize(childStats, 2 * leaf, depth + 1, dim, minMax);
                childStats = stats;
                childStats.MaxValue = std::min(childStats.MaxValue, med);
                monotonize(childStats, 2 * leaf + 1, depth + 1, dim, minMax);
            }
        }
    };

    int numDims = leafValues.ysize();
    for (int dim = 0; dim < numDims; ++dim) {
        TVector<TVector<TMinMaxStats>> minMax(numSplits + 1);
        for (int depth = 0; depth <= numSplits; ++depth)
            minMax[depth].resize(1 << depth);

        auto & lastLayerMinMax = minMax.back();
        int lastLayerSize = lastLayerMinMax.ysize();
//        std::cerr << "llz " << lastLayerSize << std::endl;
//        std::cerr << "W: ";
//        for (auto val : treeStats.LeafWeightsSum)
//            std::cerr << ' ' << val;
//        std::cerr << std::endl;
        for (int i = 0; i < lastLayerSize; ++i) {
            bool skip = treeStats.LeafWeightsSum[i] == 0;
            double value = leafValues[dim][i];
            if (skip)
                std::swap(lastLayerMinMax[i].MinValue, lastLayerMinMax[i].MaxValue);
            else
                lastLayerMinMax[i] = {value, value};
        }
        for (int depth = numSplits; depth > 0; --depth) {
            TVector<TMinMaxStats> & layerMinMax = minMax[depth - 1];
            TVector<TMinMaxStats> & prevLayerMinMax = minMax[depth];
            for (int i = 0; i < layerMinMax.ysize(); ++i) {
                layerMinMax[i] = prevLayerMinMax[2 * i];
                layerMinMax[i].update(prevLayerMinMax[2 * i + 1]);
            }
        }

//        std::cerr << "Minmax:" << std::endl;
//        int l = 0;
//        for (auto & vals : minMax)
//        {
//            std::cerr << "Layer " << l << "dim:";
//            ++l;
//            for (auto val : vals)
//                std::cerr << " (" <<  val.MinValue << ", " << val.MaxValue << ")";
//            std::cerr << std::endl;
//        }

        TMinMaxStats rootStats;
        monotonize(rootStats, 0, 0, dim, minMax);
    }
}

template <typename TError>
TVector<TVector<double>> EvalMetricPerLeafAllLayers(
    const TDataset& learnData,
    const TDataset* testData,
    TLearnContext* ctx,
    TSplitTree tree,
    const THolder<IMetric> & metric,
    const TVector<TVector<TVector<double>>> & allLayersLeafValues
) {
    int numSplits = tree.Splits.ysize();
    TVector<TVector<double>> losses(numSplits);
    while (!tree.Splits.empty())
    {
        tree.Splits.erase(tree.Splits.begin());
        --numSplits;
        int numLeafs = 1 << numSplits;
        auto indices = BuildIndices(ctx->LearnProgress.AveragingFold, tree, learnData, testData, &ctx->LocalExecutor);
        losses[numSplits] = EvalMetricPerLeaf<TError>(learnData, tree, ctx, metric, nullptr, numLeafs, indices);
    }

    return losses;
}

template <typename Error>
void PruneTreeNodes2(
    const TVector<TVector<TVector<double>>> & allLayersLeafValues,
    TVector<TVector<double>>* treeValues,
    const TVector<TVector<double>> & allLayersLosses,
    const TVector<double> & treeLosses
) {
    int numSplits = allLayersLeafValues.ysize();
    int numDims = allLayersLeafValues.at(0).ysize();

    std::function<double(int, int)> prune;
    prune = [&](int leaf, int depth) -> double {
        if (depth == numSplits)
            return treeLosses[leaf];

        int numLeafs = 1 << (numSplits - depth);
        double leftLoss = prune(2 * leaf, depth + 1);
        double rightLoss = prune(2 * leaf + 1, depth + 1);
        double curLoss = allLayersLosses[depth][leaf];
        if (leftLoss + rightLoss > curLoss) {
            int rightIdx = (leaf + 1) * numLeafs;

            for (int dim = 0; dim < numDims; ++dim) {
                double curValue = allLayersLeafValues[depth][dim][leaf];
                for (int leftIdx = leaf * numLeafs; leftIdx < rightIdx; ++leftIdx)
                    (*treeValues)[dim][leftIdx] = curValue;
            }
        } else {
            curLoss = leftLoss + rightLoss;
        }

        return curLoss;
    };

    prune(0, 0);
}

template <typename TError>
void UpdateAveragingFold(
    const TDataset& learnData,
    const TDataset* testData,
    const TError& error,
    const TSplitTree& bestSplitTree,
    TLearnContext* ctx,
    TVector<TVector<double>>* treeValues,
    TVector<TVector<double>>* prevTreeValues,
    const TVector<EMonotonicity> & monotonicFeatures
) {
    TProfileInfo& profile = ctx->Profile;
    TVector<TIndexType> indices;

//    if (!monotonicFeatures.empty())
//        SmoothApproxes<TError::StoreExpApprox>(learnData, testData, ctx);

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

    if (prevTreeValues)
    {
        for (int dim = 0; dim < treeValues->ysize(); ++dim) {
            auto & treeDim = (*treeValues)[dim];
            auto & newTreeDim = (*prevTreeValues)[dim];
            for (size_t leave = 0; leave < treeDim.ysize(); ++leave) {
                treeDim[leave] += newTreeDim[leave];
            }
        }
    }

    if (!monotonicFeatures.empty()) {
        auto treeMonotonicFeatures = GetTreeMonotonicFeatures<TError>(bestSplitTree, monotonicFeatures);
        //MonotonizeLeaveValues<TError>(treeValues, bestSplitTree, currentTreeStats, ctx, treeMonotonicFeatures);
        THolder<IMetric> metric = CreateMetric(ctx->Params.LossFunctionDescription, approxDimension);
        int numLeafs = treeValues->at(0).ysize();
        ///TVector<double> prevIterLeafsLoss = EvalMetricPerLeaf<TError>(learnData, bestSplitTree, ctx, metric, nullptr, numLeafs, indices);
        TVector<double> currIterLeafsLoss = EvalMetricPerLeaf<TError>(learnData, bestSplitTree, ctx, metric, treeValues, numLeafs, indices);
        ///PruneTreeNodes<TError>(prevIterLeafsLoss, currIterLeafsLoss, treeMonotonicFeatures, treeValues, metric);
        TVector<TVector<TVector<double>>> allLayersValues = CalcLeafValuesAllLayers(bestSplitTree, learnData, testData, error, ctx);

        std::cerr << "All layers values" << std::endl;
        for (int l = 0; l < allLayersValues.ysize(); ++l) {
            auto &layer = allLayersValues[l];
            std::cerr << "Layer " << l;
            for (auto & dim : layer) {
                std::cerr << " dim:";
                for (auto & val : dim)
                    std::cerr << ' ' << val;
                std::cerr << std::endl;
            }
        }

        std::cerr << "Leaf Leaves:" << std::endl;
        for (auto & vals : *treeValues)
        {
            std::cerr << "dim:";
            for (auto val : vals)
                std::cerr << ' '<<  val;
            std::cerr << std::endl;
        }

        MonotonizeAllLayers2<TError>(treeMonotonicFeatures, &allLayersValues, treeValues, bestSplitTree, currentTreeStats);

        std::cerr << "Monotonized all layers values" << std::endl;
        for (int l = 0; l < allLayersValues.ysize(); ++l) {
            auto &layer = allLayersValues[l];
            std::cerr << "Layer " << l;
            for (auto & dim : layer) {
                std::cerr << " dim:";
                for (auto & val : dim)
                    std::cerr << ' ' << val;
                std::cerr << std::endl;
            }
        }

        std::cerr << "Monotonized leaf Leaves:" << std::endl;
        for (auto & vals : *treeValues)
        {
            std::cerr << "dim:";
            for (auto val : vals)
                std::cerr << ' '<<  val;
            std::cerr << std::endl;
        }

        TVector<TVector<double>> allLayersLosses = EvalMetricPerLeafAllLayers<TError>(learnData, testData, ctx, bestSplitTree, metric, allLayersValues);

        std::cerr << "All layers losses" << std::endl;
        for (int l = 0; l < allLayersLosses.ysize(); ++l) {
            auto & layer = allLayersLosses[l];
            std::cerr << "Layer " << l;
            for (auto & val : layer)
                std::cerr << ' ' << val;
            std::cerr << std::endl;
        }

        PruneTreeNodes2<TError>(allLayersValues, treeValues, allLayersLosses, currIterLeafsLoss);

        std::cerr << "Tree Leaves:" << std::endl;
        for (auto & vals : *treeValues)
        {
            std::cerr << "dim: ";
            for (auto val : vals)
                std::cerr << val << ' ';
            std::cerr << std::endl;
        }

        currentTreeStats.CalcStats(*treeValues);
    }

    if (prevTreeValues)
    {
        for (int dim = 0; dim < treeValues->ysize(); ++dim) {
            auto & treeDim = (*treeValues)[dim];
            auto & newTreeDim = (*prevTreeValues)[dim];
            for (size_t leave = 0; leave < treeDim.ysize(); ++leave) {
                treeDim[leave] -= newTreeDim[leave];
            }
        }
    }

    UpdateLeafApproxes<TError::StoreExpApprox>(learnData, testData, bestSplitTree, ctx, *treeValues, indices);
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

    auto numPrevAddedTrees = ctx->LearnProgress.TreeStruct.ysize();


    TSplitTree bestSplitTree;
    const auto & monotonicFeatures = ctx->Params.DataProcessingOptions->MonotonicFeatures.Get();
    int randTreeIndex = -1;
    TVector<TVector<double>> * prevTreeLeaves = nullptr;

//    if (numPrevAddedTrees && (ctx->Rand.GenRand() % 100 < 50) ) {
//        randTreeIndex = ctx->Rand.GenRand() % numPrevAddedTrees;
//        bestSplitTree = ctx->LearnProgress.TreeStruct[randTreeIndex];
//        prevTreeLeaves = &ctx->LearnProgress.LeafValues[randTreeIndex];
//    } else
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

//    if (!monotonicFeatures.empty())
//        SiftDownMonotonicSplits<TError>(&bestSplitTree, monotonicFeatures);

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
        UpdateAveragingFold(learnData, testData, error, bestSplitTree, ctx, &treeValues, prevTreeLeaves, monotonicFeatures);

        ctx->LearnProgress.LeafValues.push_back(treeValues);
        ctx->LearnProgress.TreeStruct.push_back(bestSplitTree);

        profile.AddOperation("Update final approxes");
        CheckInterrupted(); // check after long-lasting operation

        SmoothTrees<TError>(learnData, testData, ctx);

//        auto numAddedTrees = ctx->LearnProgress.TreeStruct.ysize();
//        if (!monotonicFeatures.empty() && numAddedTrees > 1) {
//            for (int i = 0; i < 0.1 * numAddedTrees; ++i)
//            {
//                auto randTreeIndex = ctx->Rand.GenRand() % numAddedTrees;
//                const auto & randTree = ctx->LearnProgress.TreeStruct[randTreeIndex];
//                auto & randTreeValues = ctx->LearnProgress.LeafValues[randTreeIndex];
//                const auto & randTreeStats = ctx->LearnProgress.TreeStats[randTreeIndex];
//                UpdateTreeLeaves(learnData, testData, error, randTree, ctx, &randTreeValues, randTreeStats,
//                                 monotonicFeatures);
//            }
//
//            profile.AddOperation("Update random tree leaves");
//            CheckInterrupted(); // check after long-lasting operation
//        }

//        std::cerr << "Result Leaves:" << std::endl;
//        for (auto & vals : treeValues)
//        {
//            std::cerr << "dim: ";
//            for (auto val : vals)
//                std::cerr << val << ' ';
//            std::cerr << std::endl;
//        }
    }
}
