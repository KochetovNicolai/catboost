#include "train.h"
#include "train_templ.h"


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
