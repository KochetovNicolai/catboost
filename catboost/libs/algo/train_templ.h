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
void MonotonizeLeaveValues(TVector<TVector<double>>* leafValues,
                           const TSplitTree& tree,
                           TLearnContext* ctx,
                           int numMonotonicFeatures)
{
    const int approxDimension = ctx->LearnProgress.AveragingFold.GetApproxDimension();
    const int leafCount = tree.GetLeafCount();

    for (int dim = 0; dim < approxDimension; ++dim)
    {
        for (size_t shift = 1; shift < (1u << numMonotonicFeatures); shift *= 2)
        {
            for (int leaf = 0; leaf < leafCount; leaf += 2 * shift)
            {
                double mx_l = (*leafValues)[dim][leaf];
                double mn_r = (*leafValues)[dim][leaf + shift];

                for (int i = 1; i < shift; ++i)
                {
                    mx_l = std::max(mx_l, (*leafValues)[dim][leaf + i]);
                    mn_r = std::min(mn_r, (*leafValues)[dim][leaf + shift + i]);
                }

                std::cerr << "mx_l " << mx_l << " mn_r " << mn_r << std::endl;
                if (mx_l > mn_r)
                {
                    double med = 0.5 * (mx_l + mn_r);
                    for (int i = 0; i < shift; ++i)
                    {
                        auto & val_l = (*leafValues)[dim][leaf + i];
                        auto & val_r = (*leafValues)[dim][leaf + shift + i];
                        val_l = std::min(val_l, med);
                        val_r = std::max(val_r, med);
                    }
                }
            }
        }
    }

//    std::cerr << "-1 Leaves:" << std::endl;
//    for (auto & vals : *leafValues)
//    {
//        std::cerr << "dim: ";
//        for (auto val : vals)
//            std::cerr << val << ' ';
//        std::cerr << std::endl;
//    }
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
    int numMonotonicFeatures,
    bool useLearningRate
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

    if (prevTreeValues)
    {
        for (int i = 0; i < treeValues->ysize(); ++i)
        {
            auto & treeDim = (*treeValues)[i];
            auto & prevTreeDim = (*prevTreeValues)[i];
            for (size_t j = 0; j < treeDim.ysize(); ++j)
                treeDim[j] = prevTreeDim[j] + treeDim[j] * learningRate;
        }
    }

    MonotonizeLeaveValues<TError>(treeValues, bestSplitTree, ctx, numMonotonicFeatures);

    if (prevTreeValues)
    {
        for (int i = 0; i < treeValues->ysize(); ++i)
        {
            auto & treeDim = (*treeValues)[i];
            auto & prevTreeDim = (*prevTreeValues)[i];
            for (size_t j = 0; j < treeDim.ysize(); ++j)
                treeDim[j] -= prevTreeDim[j];
        }
    }

//    std::cerr << "0 Leaves:" << std::endl;
//    for (auto & vals : *treeValues)
//    {
//        std::cerr << "dim: ";
//        for (auto val : vals)
//            std::cerr << val << ' ';
//        std::cerr << std::endl;
//    }

    auto& currentTreeStats = ctx->LearnProgress.TreeStats.emplace_back();
    currentTreeStats.LeafWeightsSum.resize((*treeValues)[0].size());
    for (auto docId = 0; docId < learnData.GetSampleCount(); ++docId) {
        currentTreeStats.LeafWeightsSum[indices[ctx->LearnProgress.AveragingFold.LearnPermutation[docId]]] += learnData.Weights[docId];
    }
    // TODO(nikitxskv): if this will be a bottleneck, we can use precalculated counts.
    if (IsPairwiseError(ctx->Params.LossFunctionDescription->GetLossFunction())) {
        NormalizeLeafValues(indices, learnData.GetSampleCount(), treeValues);
    }

    TVector<TVector<double>> expTreeValues;
    expTreeValues.yresize(approxDimension);
    for (int dim = 0; dim < approxDimension; ++dim) {
        if (useLearningRate && !prevTreeValues) {
            for (auto & leafVal : (*treeValues)[dim]) {
                leafVal *= learningRate;
            }
        }
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

/// Move monotonic features to the end of list. Return the number of monotonic features.
template <typename TError>
int ReorderSplits(TSplitTree& splitTree, const TVector<int>& monotonicFeatures)
{
    THashSet<int> monotonic(monotonicFeatures.begin(), monotonicFeatures.end());

    std::cerr << "Mon: ";
    for (auto f : monotonicFeatures)
        std::cerr << f << ' ';
    std::cerr << std::endl << "Indexes: ";
    for (auto & spl : splitTree.Splits)
        std::cerr << spl.FeatureIdx << ' ';
    std::cerr << std::endl;

    int left = 0;
    int right = splitTree.Splits.ysize() - 1;
    while (left < right)
    {
        while (left < splitTree.Splits.ysize() && !monotonic.has(splitTree.Splits[left].FeatureIdx))
            ++left;
        while (right >=0 && monotonic.has(splitTree.Splits[right].FeatureIdx))
            --right;
        if (left < right)
        {
            std::swap(splitTree.Splits[left], splitTree.Splits[right]);
            ++left;
            --right;
        }
    }
    return splitTree.Splits.ysize() - left;
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

    int numMonotonicFeatures = ReorderSplits<TError>(bestSplitTree, ctx->Params.DataProcessingOptions->MonotonicFeatures.Get());
    std::cerr << "numMonotonicFeatures: " << numMonotonicFeatures << std::endl;

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
        UpdateAveragingFold(learnData, testData, error, bestSplitTree, ctx, &treeValues, nullptr, numMonotonicFeatures, true);

        ctx->LearnProgress.LeafValues.push_back(treeValues);
        ctx->LearnProgress.TreeStruct.push_back(bestSplitTree);
        ctx->LearnProgress.NumMonotonicFeatures.push_back(numMonotonicFeatures);

        profile.AddOperation("Update final approxes");
        CheckInterrupted(); // check after long-lasting operation

        if (ctx->LearnProgress.TreeStruct.ysize() > 1)
        {
            auto randTreeIndex = ctx->Rand.GenRand() % (ctx->LearnProgress.TreeStruct.ysize() - 1);
            //for (int randTreeIndex = 0; randTreeIndex < ctx->LearnProgress.TreeStruct.ysize() - 1; ++randTreeIndex)
            {
                auto & randTree = ctx->LearnProgress.TreeStruct[randTreeIndex];
                auto & randTreeValues = ctx->LearnProgress.LeafValues[randTreeIndex];
                auto randTreeNumMonotonicFeatures = ctx->LearnProgress.NumMonotonicFeatures[randTreeIndex];
                TVector<TVector<double>> randTreeAdditionalValues; // [dim][leafId]

                std::cerr << "Prev Leaves:" << std::endl;
                for (auto & vals : randTreeValues)
                {
                    std::cerr << "dim: ";
                    for (auto val : vals)
                        std::cerr << val << ' ';
                    std::cerr << std::endl;
                }

                UpdateAveragingFold(learnData, testData, error, randTree, ctx, &randTreeAdditionalValues, &randTreeValues, randTreeNumMonotonicFeatures, true);

                std::cerr << "Fixed Leaves:" << std::endl;
                for (auto & vals : randTreeAdditionalValues)
                {
                    std::cerr << "dim: ";
                    for (auto val : vals)
                        std::cerr << val << ' ';
                    std::cerr << std::endl;
                }

                for (int i = 0; i < randTreeValues.ysize(); ++i)
                {
                    auto & treeDim = (randTreeValues)[i];
                    auto & prevTreeDim = (randTreeAdditionalValues)[i];
                    for (size_t j = 0; j < treeDim.ysize(); ++j)
                        treeDim[j] += prevTreeDim[j];
                }
            }
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
