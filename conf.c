/*
*
* This file contains some hard-coded information about the tables and the IMMVs.
* All these information can be extracted from the system catalog,
* but for the sake of simplicity, we hard-code them here.
*
*
**/



#include "postgres.h"
#include "conf.h"

const Oid immv_on_nations[] = { tpch_5, tpch_7, tpch_9, tpch_10 };
const Oid immv_on_regions[] = { tpch_5 };
const Oid immv_on_customers[] = { tpch_3, tpch_5, tpch_7, tpch_10 };
const Oid immv_on_orders[] = { tpch_3, tpch_5, tpch_7, tpch_9, tpch_10, tpch_12 };
const Oid immv_on_lineitem[] = {
	tpch_1, tpch_3, tpch_5, tpch_7, tpch_9, tpch_10, tpch_12, tpch_19
};
const Oid immv_on_part[] = { tpch_9, tpch_19 };
const Oid immv_on_partsupp[] = { tpch_9 };
const Oid immv_on_supplier[] = { tpch_5, tpch_7, tpch_9 };

const Oid all_immvs[] = { tpch_1, tpch_3, tpch_5, tpch_7, tpch_9, tpch_10, tpch_12, tpch_19 };

RefedImmv refed_immv[8] = {
	{ nation, immv_on_nations, "nation", sizeof(immv_on_nations) / sizeof(Oid) },
	{ region, immv_on_regions, "region", sizeof(immv_on_regions) / sizeof(Oid) },
	{ customer, immv_on_customers, "customer", sizeof(immv_on_customers) / sizeof(Oid) },
	{ orders, immv_on_orders, "orders", sizeof(immv_on_orders) / sizeof(Oid) },
	{ lineitem, immv_on_lineitem, "lineitem", sizeof(immv_on_lineitem) / sizeof(Oid) },
	{ part, immv_on_part, "part", sizeof(immv_on_part) / sizeof(Oid) },
	{ partsupp, immv_on_partsupp, "partsupp", sizeof(immv_on_partsupp) / sizeof(Oid) },
	{ supplier, immv_on_supplier, "supplier", sizeof(immv_on_supplier) / sizeof(Oid) }
};

RefedImmv *
getRefrenceImmv(Oid relOid)
{
	int i;
	for (i = 0; i < 8; i++)
	{
		if (refed_immv[i].relOid == relOid)
			return &refed_immv[i];
	}
	return NULL;
}

/* Map the Table oid into a sequence starting with 0 */
int getIndexForTableEmbeeding(Oid oid){
	if (oid == customer)
		return 1;
	else if (oid == lineitem)
		return 2;
	else if (oid == nation)
		return 3;
	else if (oid == orders)
		return 4;
	else if (oid == part)
		return 5;
	else if (oid == partsupp)
		return 6;
	else if (oid == region)
		return 7;
	else if (oid == supplier)
		return 8;
	else if (oid == tpch_1)
		return 9;
	else if (oid == tpch_3)
		return 10;
	else if (oid == tpch_5)
		return 11;
	else if (oid == tpch_7)
		return 12;
	else if (oid == tpch_9)
		return 13;
	else if (oid == tpch_10)
		return 14;
	else if (oid == tpch_12)
		return 15;
	else if (oid == tpch_19)
		return 16;
	else
		return 0;
}
