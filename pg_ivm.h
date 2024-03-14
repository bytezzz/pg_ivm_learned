/*-------------------------------------------------------------------------
 *
 * pg_ivm.h
 *	  incremental view maintenance extension
 *
 * Portions Copyright (c) 1996-2022, PostgreSQL Global Development Group
 * Portions Copyright (c) 2022, IVM Development Group
 *
 *-------------------------------------------------------------------------
 */

#ifndef _PG_IVM_H_
#define _PG_IVM_H_

#include "catalog/objectaddress.h"
#include "fmgr.h"
#include "nodes/params.h"
#include "nodes/pathnodes.h"
#include "parser/parse_node.h"
#include "tcop/dest.h"
#include "utils/queryenvironment.h"
#include "storage/lwlock.h"
#include "access/transam.h"
#include "utils/elog.h"
#include "nodes/plannodes.h"
#include "utils/hsearch.h"
#include "executor/execdesc.h"
#include "cJSON.h"
#include <zmq.h>

#define Natts_pg_ivm_immv 3

#define Anum_pg_ivm_immv_immvrelid 1
#define Anum_pg_ivm_immv_viewdef 2
#define Anum_pg_ivm_immv_ispopulated 3

/* pg_ivm.c */

extern void CreateChangePreventTrigger(Oid matviewOid);
extern Oid PgIvmImmvRelationId(void);
extern Oid PgIvmImmvPrimaryKeyIndexId(void);
extern bool isImmv(Oid immv_oid);

/* createas.c */

extern ObjectAddress ExecCreateImmv(ParseState *pstate, CreateTableAsStmt *stmt,
									ParamListInfo params, QueryEnvironment *queryEnv,
									QueryCompletion *qc);
extern void CreateIvmTriggersOnBaseTables(Query *qry, Oid matviewOid);
extern void CreateIndexOnIMMV(Query *query, Relation matviewRel);
extern Query *rewriteQueryForIMMV(Query *query, List *colNames);
extern void makeIvmAggColumn(ParseState *pstate, Aggref *aggref, char *resname,
							 AttrNumber *next_resno, List **aggs);

/* matview.c */

extern Query *get_immv_query(Relation matviewRel);
extern ObjectAddress ExecRefreshImmv(const RangeVar *relation, bool skipData,
									 const char *queryString, QueryCompletion *qc);
extern bool ImmvIncrementalMaintenanceIsEnabled(void);
extern Query *get_immv_query(Relation matviewRel);
extern Datum IVM_immediate_before(PG_FUNCTION_ARGS);
extern Datum IVM_immediate_maintenance(PG_FUNCTION_ARGS);
extern Query *rewrite_query_for_exists_subquery(Query *query);
extern Datum ivm_visible_in_prestate(PG_FUNCTION_ARGS);
extern void AtAbort_IVM(void);
extern char *getColumnNameStartWith(RangeTblEntry *rte, char *str, int *attnum);
extern bool isIvmName(const char *s);

/* ruleutils.c */

extern char *pg_ivm_get_viewdef(Relation immvrel, bool pretty);

/* subselect.c */
extern void inline_cte(PlannerInfo *root, CommonTableExpr *cte);

/* Learned_ivm-related Structures */

/* Decision server info */
#define SERVERNAME  "localhost"
#define PORT  2300

/*ivm related log level*/
#define IVM_LOG_LEVEL LOG

enum query_staus
{
	/* After a new received query logging into the global structure, it will become blocked*/
	QUERY_BLOCKED,

	/* Reschedule functions will change their status into "avaliable",
	 * after which corresponding backend process can execute them. */
	QUERY_AVAILABLE,

	/* If a query can not acquire all necessary locks, it will give up this schedule,
		queries in the status of giving up can not be start in next schedule decision,
		but will become avaliable after a schedule event.*/
	QUERY_GIVE_UP
};

typedef int ScheduleTag;

enum schedule_tag
{
	QUERY_FINISHED,

	QUERY_GIVEUP,

	INCOMING_QUERY,

	DEAD_WAKEUP,
};

/* Configurable parameters */
#define MAX_QUERY_NUM 1000
#define MAX_QUERY_LENGTH ((Size) 8192)
#define MAX_AFFECTED_TABLE 100

#define MAX_CONCURRENT_QUERY 4

#define QUERY_EMBEDDING_SIZE 1024

#define QUERY_PLAN_JSON_SIZE 4096

//8 base tables, 8 immvs, 1 for all other tables
#define WORKING_TABLES 17

#define HASH_TABLE_SIZE (MAX_QUERY_NUM * sizeof(QueryTableEntry) + 32 * sizeof(LockedTableEntry))

/*
 * The current implementation heavily depends on shared memory,
 * however, the size of shared memory is fixed after allocation.
 * Therefore, we need to use fixed size for all the data structures
 * which is not scalable.
 * Maybe switch to the new dynamic shared memory features in fulture.
 */

typedef struct QueryTableKey
{
	char query_string[MAX_QUERY_LENGTH];
	uint32 pid;
} QueryTableKey;

typedef struct QueryTableEntry
{
	QueryTableKey key;
	Oid affected_tables[MAX_AFFECTED_TABLE];
	char query_plan_json[QUERY_PLAN_JSON_SIZE];
	int status;
	TransactionId xid;
	clock_t start_time;
	int outside_finished;
	int num_of_relation_access;

	/* decision id if waken up by the server, -1 otherwise */
	int wakeup_by;
} QueryTableEntry;

typedef struct LockedRelationKey
{
	Oid table;
} LockedTable;

typedef struct LockedRelationEntry
{
	LockedTable key;
	TransactionId held_by;
} LockedTableEntry;

/* Saving all necessary information we need for query scheduling*/
typedef struct SchedueState
{
	int querynum;
	int runningQuery;

	/* We only need one lock here, since the change to QueryTable
				will subsequently affect ScheduleTable
	*/
	LWLock *lock;

	int tableAccessCounter[WORKING_TABLES];
} ScheduleState;

#pragma pack(1)

typedef struct payload_t
{
	double embedding[1024];
} query_embed;

typedef struct response_t
{
	uint32_t decision_id;
	uint32_t decision;
} response;

typedef struct env_features_t
{
	int schedule_tag;
	int wakeup_decision_id;
	int LRU[WORKING_TABLES];
} env_features;

// A struct to represent a query plan before we transform it into JSON.
typedef struct BaoPlanNode {
  // An integer representation of the PG NodeTag.
  unsigned int node_type;

  // The optimizer cost for this node (total cost).
  double optimizer_cost;

  // The cardinality estimate (plan rows) for this node.
  double cardinality_estimate;

  // If this is a scan or index lookup, the name of the underlying relation.
  char* relation_name;

  // Left child.
  struct BaoPlanNode* left;

  // Right child.
  struct BaoPlanNode* right;
} BaoPlanNode;

#pragma pack()

#define SEGMENT_SIZE (sizeof(ScheduleState))

/* querysched.c */

extern void LogQuery(ScheduleState *state, QueryDesc *desc, QueryTableEntry *query_entry);
extern void Reschedule(HTAB *queryTable, ScheduleState *state, ScheduleTag tag, env_features *env);

extern void
UpdateStateForRemoving(ScheduleState *state, QueryDesc *desc, QueryTableEntry *query_entry, bool is_running);

/* env_embedding.c */

extern void log_table_access(ScheduleState * ss, Oid *affected_table);


/* utils.c */


typedef int HashQueryType;

extern QueryTableEntry *GetEntry(HTAB *table, QueryDesc *queryDesc, HASHACTION type, bool* found);
extern int connect_to_server(const char* host, int port);
extern int send_msg(int sock, const char *msg, uint32_t msgsize);
extern void write_json_to_socket(int conn_fd, const char* json);


extern char* plan_to_json(PlannedStmt* plan);
extern cJSON* env_to_json(env_features* env);

extern cJSON* buffer_state();

#endif
