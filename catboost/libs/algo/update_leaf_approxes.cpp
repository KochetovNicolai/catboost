#include "train.h"
#include "train_templ.h"

template <bool, bool>
void UpdateLeafApproxes(const TDataset & learnData,
                        const TDataset * testData,
                        const TSplitTree & bestSplitTree,
                        TLearnContext * ctx,
                        TVector<TVector<double>> * treeValues,
                        const TVector<TIndexType> & indices);

TUpdateLeafApproxesFunction GetUpdateLeafApproxesFunction(bool StoreExpApprox, bool RollbackTree)
{
    if (StoreExpApprox)
    {
        if (RollbackTree)
            return UpdateLeafApproxes<true, true>;
        else
            return UpdateLeafApproxes<true, false>;
    } else {
        if (RollbackTree)
            return UpdateLeafApproxes<false, true>;
        else
            return UpdateLeafApproxes<false, false>;
    }
}
