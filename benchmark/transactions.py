import psycopg
import random

CURRENT_DATE = '1995-06-17'

PRIORITY = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW']

SHIPINSTRUCT = ['DELIVER IN PERSON', 'COLLECT COD', 'NONE', 'TAKE BACK RETURN']

SHIPMODE = ['AIR', 'SHIP', 'RAIL', 'TRUCK', 'MAIL']

def get_connection():
    return psycopg.connect("host=127.0.0.1 dbname=tpch user=vscode password=vscode port=5432")

def shipment(conn: psycopg.Connection):
  cursor = conn.cursor()
  line = cursor.execute("SELECT l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity FROM lineitem WHERE l_linestatus='O' ORDER BY RANDOM() LIMIT 1;").fetchone()
  order_key, part_key, supp_key, line_number, quantity = line
  ava_quantity = cursor.execute(f'Select ps_availqty from partsupp where ps_partkey = {part_key} and ps_suppkey = {supp_key};').fetchone()[0]

  #update l_shipdate and l_linestatus
  cursor.execute(f'UPDATE lineitem SET l_shipdate = date \'{CURRENT_DATE}\', l_linestatus = \'F\' WHERE l_orderkey = {order_key} and l_linenumber = {line_number};')

  if ava_quantity < quantity:
  #add avaliable quantity by 1.5 * quantity
    cursor.execute(f'UPDATE partsupp SET ps_availqty = ps_availqty + {1.5 * quantity} WHERE ps_partkey = {part_key} and ps_suppkey = {supp_key};')
  # decrease avaliable quantity by quantity
  cursor.execute(f'UPDATE partsupp SET ps_availqty = ps_availqty - {quantity} WHERE ps_partkey = {part_key} and ps_suppkey = {supp_key};')



  #check if we need to update order table
  if cursor.execute(f'SELECT COUNT(*) FROM lineitem WHERE l_orderkey = {order_key} and l_linestatus = \'O\' and l_linenumber <> {line_number};').fetchone()[0] == 0:
  #update order table
    cursor.execute(f'UPDATE orders SET o_orderstatus = \'F\' WHERE o_orderkey = {order_key};')

  conn.commit()

def good_receive(conn: psycopg.Connection):
  cursor = conn.cursor()
  order_key, linenumber = cursor.execute(f"select l_orderkey, l_linenumber from lineitem where l_linestatus = 'F' and l_returnflag = 'N' and l_shipdate < date '{CURRENT_DATE}' order by RANDOM() limit 1;").fetchone()
  #update lineitem set l_shipdate = date '1995-06-17', l_returnflag = random.choice(['A','R']) where l_orderkey = , l_linenumber = ;
  cursor.execute(f"update lineitem set l_shipdate = date '{CURRENT_DATE}', l_returnflag = \'{random.choice(['A','R'])}\' where l_orderkey = {order_key} and l_linenumber = {linenumber};")
  conn.commit()

def new_order(conn: psycopg.Connection):
  cursor = conn.cursor()
  cuskey, nationkey, acctbal = cursor.execute("SELECT c_custkey, c_nationkey, c_acctbal from customer ORDER by RANDOM() limit 1;").fetchone()

  temp_table_generation = """
    SELECT
      ps_partkey AS partkey,
      ps_suppkey AS suppkey,
      p.p_retailprice AS retailprice,
      {quantity} AS quantity,
      {tax} AS tax,
      {discont} as discount,
      {returnflag} as returnflag,
      {linestatus} as linestatus,
      p.p_retailprice * {quantity} AS extented_price,
      row_number() over (ORDER BY RANDOM()) as linenumber INTO temp
    FROM
      partsupp ps,
      supplier s,
      part p
    WHERE
      p.p_partkey = ps.ps_partkey
      AND ps.ps_suppkey = s.s_suppkey
      AND s.s_suppkey in (
        SELECT
          s_suppkey FROM supplier
        WHERE
          s_nationkey in (
            SELECT
              n_nationkey FROM nation
            WHERE
              n_regionkey = (
                SELECT
                  n_regionkey FROM nation
                WHERE
                  n_nationkey = {nationkey})))
    ORDER BY
      RANDOM()
    LIMIT {buy_count};
  """

  discont = random.randint(0,10)/100
  tax = random.randint(0,8)/100
  conn.execute(
    temp_table_generation.format(
      quantity = random.randint(1,50),
      tax = tax,
      discont = discont,
      returnflag = '\'N\'',
      linestatus = '\'O\'',
      buy_count = random.randint(1,7),
      nationkey = nationkey
    )
  )

  temp_lineitem = """
    INSERT INTO ORDERS (O_TOTALPRICE, O_ORDERDATE, O_ORDERPRIORITY, O_CLERK, O_SHIPPRIORITY, O_COMMENT, O_ORDERKEY, O_ORDERSTATUS, O_CUSTKEY)
    SELECT
        SUM(temp.extented_price * (1+temp.tax) * (1-temp.discount)) as total_price,
        DATE '1995-06-17' - INTERVAL '{daybefore}' DAY,
        '{orderpriority}',
        '{clerk}',
        0,
        '{comment}',
        (SELECT MAX(orders.o_orderkey) +1 FROM orders),
        'O',
        {customerkey}
    FROM TEMP
    RETURNING O_ORDERKEY, O_TOTALPRICE, O_ORDERDATE;
  """

  order_key, total_price , order_date = conn.execute(
    temp_lineitem.format(
      daybefore = random.randint(1,121),
      clerk = f'Clerk#{random.randint(1,1000)}',
      comment = f'Comment#{random.randint(1,1000)}',
      customerkey = cuskey,
      orderpriority = random.choice(PRIORITY)
    )
  ).fetchone()

  if acctbal < total_price:
    #add acctbal by 1.5 * total_price
    cursor.execute(f'UPDATE customer SET c_acctbal = c_acctbal + {1.5 * float(total_price)} WHERE c_custkey = {cuskey};')

  # decrease acctbal by total_price
  cursor.execute(f'UPDATE customer SET c_acctbal = c_acctbal - {float(total_price)} WHERE c_custkey = {cuskey};')

  # Your previous code remains unchanged.

  # Insert data into LINEITEM table
  insert_into_lineitem = """
    INSERT INTO LINEITEM (
        L_ORDERKEY,
        L_PARTKEY,
        L_SUPPKEY,
        L_LINENUMBER,
        L_QUANTITY,
        L_EXTENDEDPRICE,
        L_DISCOUNT,
        L_TAX,
        L_RETURNFLAG,
        L_LINESTATUS,
        L_SHIPDATE,
        L_COMMITDATE,
        L_RECEIPTDATE,
        L_SHIPINSTRUCT,
        L_SHIPMODE,
        L_COMMENT
    )
    SELECT
        {orderkey},
        temp.partkey,
        temp.suppkey,
        temp.linenumber,
        temp.quantity,
        temp.extented_price,
        temp.discount,
        temp.tax,
        temp.returnflag,
        temp.linestatus,
        DATE {shipdate},
        DATE {commitdate},
        DATE {receiptdate},
        '{shipinstruct}',
        '{shipmode}',
        'Regular order'
    FROM TEMP;
  """

  # Assuming `order_key` is a tuple containing a single value.
  conn.execute(
    insert_into_lineitem.format(
      orderkey = order_key,
      orderdate = order_date,
      shipdate = f'\'{CURRENT_DATE}\' + INTERVAL \'{random.randint(1,121)}\' DAY',
      commitdate = f'\'{CURRENT_DATE}\' + INTERVAL \'{random.randint(1,121)}\' DAY',
      receiptdate = f'\'{CURRENT_DATE}\' + INTERVAL \'{random.randint(121,300)}\' DAY',
      shipmode = random.choice(SHIPMODE),
      shipinstruct = random.choice(SHIPINSTRUCT)
    )
  )

  # Cleaning up the temporary table after its use.
  conn.execute("DROP TABLE TEMP;")
  conn.commit();

def update_score(conn: psycopg.Connection):
  transaction = """
    WITH INFO AS (
      SELECT
        l_suppkey AS supplier_no,
        SUM(l_extendedprice * (1 - l_discount)) AS total_revenue,
        RANK() OVER (ORDER BY SUM(l_extendedprice * (1 - l_discount))
          DESC) AS revenue_rank,
        COUNT(*) FILTER (WHERE l_returnflag = 'R') AS returned_items,
        COUNT(*) AS total_items FROM lineitem
      GROUP BY
        l_suppkey
    ),
    SupplierRanks AS (
    SELECT
      supplier_no,
      revenue_rank,
      returned_items::float / total_items AS return_rate,
      total_revenue,
      RANK() OVER (ORDER BY returned_items::float / total_items) AS return_rank FROM INFO
    ),
    SupplierCount AS (
    SELECT
      COUNT(DISTINCT l_suppkey) AS total_supplier FROM lineitem
    ),
    Scores AS (
    SELECT
      sr.*,
      CASE WHEN sr.revenue_rank <= sc.total_supplier * 0.10 THEN
        4
      WHEN sr.revenue_rank <= sc.total_supplier * 0.30 THEN
        3
      WHEN sr.revenue_rank <= sc.total_supplier * 0.50 THEN
        2
      ELSE
        1
      END AS revenue_score,
      CASE WHEN sr.return_rank <= sc.total_supplier * 0.10 THEN
        4
      WHEN sr.return_rank <= sc.total_supplier * 0.30 THEN
        3
      WHEN sr.return_rank <= sc.total_supplier * 0.50 THEN
        2
      ELSE
        1
      END AS return_score FROM SupplierRanks sr,
      SupplierCount sc
    )
    UPDATE
      supplier SET
        s_comment = concat('Supplier_Score: ', (s.revenue_score::float + s.return_score) / 2)
      FROM
        Scores s
      WHERE
        supplier.s_suppkey = s.supplier_no AND RANDOM() < 0.2;
  """
  conn.execute(transaction)
  conn.commit()

def adjust_discount(conn:psycopg.Connection):
  sql_query = """
    UPDATE LINEITEM
    SET L_DISCOUNT = L_DISCOUNT * (1 + (RANDOM() - 0.5) / 10) -- adjust by up to +/- 5%
    WHERE RANDOM() < 0.001; -- assuming we're adjusting 1% of the items
  """
  cursor = conn.cursor()
  cursor.execute(sql_query)
  conn.commit()
