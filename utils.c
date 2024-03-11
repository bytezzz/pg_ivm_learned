#include "postgres.h"
#include "storage/lwlock.h"
#include "utils/elog.h"
#include "stdlib.h"
#include "miscadmin.h"
#include "parser/parsetree.h"
#include "access/xact.h"
#include "nodes/plannodes.h"
#include "utils/builtins.h"
#include "storage/lmgr.h"
#include "nodes/nodes.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>

#include "pg_ivm.h"

QueryTableEntry *GetEntry(HTAB *table, QueryDesc *queryDesc, HASHACTION type, bool *found){
	QueryTableKey key;

	memset(&key, 0, sizeof(QueryTableKey));
	key.pid = MyProcPid;
	strcpy(key.query_string, queryDesc->sourceText);

	return hash_search(table, &key, type, found);
}



int connect_to_server(const char* host, int port){
	struct sockaddr_in server_address;
	int sock = socket(AF_INET, SOCK_STREAM, 0);

	if (sock < 0)
	{
		printf("ERROR: Socket creation failed\n");
		ereport(ERROR, (errcode(ERRCODE_CONNECTION_FAILURE), errmsg("Socket creation failed")));
	}

	memset(&server_address, 0, sizeof(server_address));
	server_address.sin_family = AF_INET;
	inet_pton(AF_INET, SERVERNAME, &server_address.sin_addr);
	server_address.sin_port = htons(PORT);

	if (connect(sock,
				(struct sockaddr *) &server_address,
				sizeof(server_address)) < 0)
	{
		printf("ERROR: Unable to connect to server\n");
		ereport(ERROR,
				(errcode(ERRCODE_CONNECTION_FAILURE), errmsg("Unable to connect to server")));
	}

	return sock;
}

/* Send DataStructure through socket. */
int
send_msg(int sock, const char *msg, uint32_t msgsize)
{
	int sent = write(sock, msg, msgsize);
	if (sent < 0)
	{
		ereport(ERROR, (errcode(ERRCODE_IO_ERROR), errmsg("Can't send message.")));
		close(sock);
	}
	return sent;
}

void write_json_to_socket(int conn_fd, const char* json) {
  size_t json_length;
  ssize_t written, written_total;
  json_length = strlen(json);
  written_total = 0;

  while (written_total != json_length) {
    written = send_msg(conn_fd,
                    json + written_total,
                    json_length - written_total);
    written_total += written;
  }
}

