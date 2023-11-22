# contrib/pg_ivm/Makefile


MODULE_big = pg_ivm
OBJS = \
	$(WIN32RES) \
	createas.o \
	matview.o \
	pg_ivm.o \
	ruleutils.o \
	subselect.o \
	querysched.o
PGFILEDESC = "pg_ivm - incremental view maintenance on PostgreSQL"

EXTENSION = pg_ivm
DATA = pg_ivm--1.0.sql \
       pg_ivm--1.0--1.1.sql pg_ivm--1.1--1.2.sql pg_ivm--1.2--1.3.sql \
       pg_ivm--1.3--1.4.sql pg_ivm--1.4--1.5.sql pg_ivm--1.5--1.6.sql \
       pg_ivm--1.6--1.7.sql

REGRESS = pg_ivm create_immv refresh_immv

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Define the path variable
PG_PATH ?= /home/vscode/pgsql
LOG_FILE ?= /home/vscode/logfile

stop:
	pg_ctl -D $(PG_PATH) -l $(LOG_FILE) stop

start:
	pg_ctl -D $(PG_PATH) -l $(LOG_FILE) start

restart: stop start

.PHNOY: benchmark

benchmark:
	python3 benchmark/concurrent_test.py

kill:
	@PID=$$(ps -ef | grep '[p]ostgres -D' | awk '{print $$2}') ; \
	if [ -n "$$PID" ]; then \
		pkill -9 -P $$PID ; \
		echo "Killed all children processes of PostgreSQL with PID $$PID" ; \
	else \
		echo "No main PostgreSQL process found." ; \
	fi

