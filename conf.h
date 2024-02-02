#include "postgres.h"

#ifndef _IVM_CONFIG_H_

#define _IVM_CONFIG_H_

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
#define immv_count 8



typedef struct RefedImmv
{
	Oid relOid;
	const Oid *refed_table;
	const char *refed_table_name;
	int refed_table_num;
} RefedImmv;

extern RefedImmv *getRefrenceImmv(Oid relOid);
extern int getIndexForTableEmbeeding(Oid oid);
extern const Oid immv_on_nations[];
extern const Oid immv_on_regions[];
extern const Oid immv_on_customers[];
extern const Oid immv_on_orders[];
extern const Oid immv_on_lineitem[];
extern const Oid immv_on_part[];
extern const Oid immv_on_partsupp[];
extern const Oid immv_on_supplier[];
extern const Oid all_immvs[];

extern RefedImmv refed_immv[];

#endif

