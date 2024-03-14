#include "postgres.h"
#include "stdlib.h"
#include "stdio.h"
#include "pg_ivm.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "parser/parsetree.h"


static char* get_relation_name(PlannedStmt* stmt, Plan* node) {
  Index rti;

  switch (node->type) {
  case T_SeqScan:
  case T_SampleScan:
  case T_IndexScan:
  case T_IndexOnlyScan:
  case T_BitmapHeapScan:
  case T_BitmapIndexScan:
  case T_TidScan:
  case T_ForeignScan:
  case T_CustomScan:
    rti = ((Scan*)node)->scanrelid;
    return get_rel_name(rt_fetch(rti, stmt->rtable)->relid);
    break;
  case T_ModifyTable:
    rti = ((ModifyTable*)node)->nominalRelation;
    return get_rel_name(rt_fetch(rti, stmt->rtable)->relid);
    break;
  default:
    return NULL;
  }
}

// Transform the operator types we care about from their PG tag to a
// string. Call other operators "Other".
static const char* node_type_to_string(NodeTag tag) {
  switch (tag) {
  case T_SeqScan:
    return "Seq Scan";
  case T_IndexScan:
    return "Index Scan";
  case T_IndexOnlyScan:
    return "Index Only Scan";
  case T_BitmapIndexScan:
    return "Bitmap Index Scan";
  case T_NestLoop:
    return "Nested Loop";
  case T_MergeJoin:
    return "Merge Join";
  case T_HashJoin:
    return "Hash Join";
  default:
    return "Other";
  }
}

// Allocate an empty BaoPlanNode.
static BaoPlanNode* new_bao_plan() {
  return (BaoPlanNode*) malloc(sizeof(BaoPlanNode));
}

// Free (recursively) an entire BaoPlanNode. Frees children as well.
static void free_bao_plan_node(BaoPlanNode* node) {
  if (node->left) free_bao_plan_node(node->left);
  if (node->right) free_bao_plan_node(node->right);
  free(node);
}

static cJSON* emit_json(BaoPlanNode* node) {
  cJSON* json = cJSON_CreateObject();
  cJSON_AddStringToObject(json, "Node Type", node_type_to_string(node->node_type));
  cJSON_AddNumberToObject(json, "Node Type ID", node->node_type);
  if (node->relation_name)
    cJSON_AddStringToObject(json, "Relation Name", node->relation_name);
  cJSON_AddNumberToObject(json, "Total Cost", node->optimizer_cost);
  cJSON_AddNumberToObject(json, "Plan Rows", node->cardinality_estimate);
  if (node->left || node->right) {
    cJSON* plans = cJSON_CreateArray();
    if (node->left) cJSON_AddItemToArray(plans, emit_json(node->left));
    if (node->right) cJSON_AddItemToArray(plans, emit_json(node->right));
    cJSON_AddItemToObject(json, "Plans", plans);
  }
  return json;
}

// Transform a PostgreSQL PlannedStmt into a BaoPlanNode tree.
static BaoPlanNode* transform_plan(PlannedStmt* stmt, Plan* node) {
  BaoPlanNode* result = new_bao_plan();

  result->node_type = node->type;
  result->optimizer_cost = node->total_cost;
  result->cardinality_estimate = node->plan_rows;
  result->relation_name = get_relation_name(stmt, node);

  result->left = NULL;
  result->right = NULL;
  if (node->lefttree) result->left = transform_plan(stmt, node->lefttree);
  if (node->righttree) result->right = transform_plan(stmt, node->righttree);

  return result;
}

// Given a PostgreSQL PlannedStmt, produce the JSON representation we need to
// send to the Bao server.
char* plan_to_json(PlannedStmt* plan) {
  BaoPlanNode* transformed_plan;
  cJSON* json;
  char *json_str;

  transformed_plan = transform_plan(plan, plan->planTree);

  json = cJSON_CreateObject();
  cJSON_AddItemToObject(json, "Plan", emit_json(transformed_plan));
  cJSON_AddItemToObject(json, "Buffers", buffer_state());

  free_bao_plan_node(transformed_plan);
  json_str = cJSON_Print(json);
  cJSON_free(json);

  return json_str;
}

cJSON* env_to_json(env_features* env) {
  cJSON* array;
  cJSON* json = cJSON_CreateObject();
  array = cJSON_CreateIntArray(env->LRU, WORKING_TABLES);
  cJSON_AddNumberToObject(json, "schedule_tag", env->schedule_tag);
  cJSON_AddNumberToObject(json, "wakeup_decision_id", env->wakeup_decision_id);
  cJSON_AddItemToObject(json, "LRU", array);

  return json;
}
