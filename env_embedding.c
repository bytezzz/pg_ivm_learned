#include "conf.h"
#include "pg_ivm.h"


void log_table_access(ScheduleState * ss, Oid *affected_table){

  int i;
  bool accessed[WORKING_TABLES] = {0};

	for (i = 0; i < MAX_AFFECTED_TABLE && affected_table[i] != 0; i++){
    accessed[getIndexForTableEmbeeding(affected_table[i])] = true;
  }

  for (i = 0; i < WORKING_TABLES; i++){
    if (accessed[i]){
      ss->LRUTableAccess[i] = 0;
    }else{
      if (ss->LRUTableAccess[i] < 10)
        ss->LRUTableAccess[i]++;
    }
  }
}
