#include "postgres.h"

#define tpch_1 39992
#define tpch_3 40011
#define tpch_5 40046
#define tpch_7 40105
#define tpch_9 40156
#define tpch_10 40215
#define tpch_12 40258
#define tpch_19 40283
#define nation 16390
#define region 16393
#define customer 16405
#define orders 16408
#define lineitem 16411
#define part 16396
#define partsupp 16402
#define supplier 16399



typedef struct RefedImmv
{
	Oid relOid;
	const Oid *refed_table;
	const char *refed_table_name;
	int refed_table_num;
} RefedImmv;

RefedImmv *getRefrenceImmv(Oid relOid);
int getIndexForTableEmbeeding(Oid oid);
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
int immv_count = 8;

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

int getIndexForTableEmbeeding(Oid oid){
	if (oid == customer)
		return 0;
	else if (oid == lineitem)
		return 1;
	else if (oid == nation)
		return 2;
	else if (oid == orders)
		return 3;
	else if (oid == part)
		return 4;
	else if (oid == partsupp)
		return 5;
	else if (oid == region)
		return 6;
	else if (oid == supplier)
		return 7;
	else
		return -1;
}
