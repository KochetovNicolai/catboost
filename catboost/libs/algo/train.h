#pragma once

#include "learn_context.h"
#include <catboost/libs/overfitting_detector/error_tracker.h>

using TTrainOneIterationFunc = std::function<void(const TDataset& learnData,
                                                  const TDataset* testData,
                                                  TLearnContext* ctx)>;

TTrainOneIterationFunc GetOneIterationFunc(ELossFunction lossFunction);

using TUpdateLeafApproxesFunction = std::function<void(const TDataset & learnData,
                                                       const TDataset * testData,
                                                       const TSplitTree & bestSplitTree,
                                                       TLearnContext * ctx,
                                                       const TVector<TVector<double>> & treeValues,
                                                       const TVector<TIndexType> & indices)>;

TUpdateLeafApproxesFunction GetUpdateLeafApproxesFunction(bool StoreExpApprox, bool RollbackTree);

TErrorTracker BuildErrorTracker(EMetricBestValue bestValueType, float bestPossibleValue, bool hasTest, TLearnContext* ctx);
