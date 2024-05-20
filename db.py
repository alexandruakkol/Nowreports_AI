from pymilvus import Collection, connections, utility, FieldSchema, DataType, CollectionSchema, AnnSearchRequest, RRFRanker, MilvusClient
from dotenv import load_dotenv
import pg8000.native
import os
import numpy as np
import sys

load_dotenv()

connections.connect(
  alias=os.getenv('MVALIAS'),
  user=os.getenv('MVUSER'),
  password=os.getenv('MVPASS'),
  host=os.getenv('MVHOST'),
  port=os.getenv('MVPORT')
)

print("Connection to Milvus established successfully.")

try:
  collection = Collection('nowreports_beta') # nowreports_bge | test | nowreports_beta
  print(collection)
  collection.load()
except Exception as e:
  print(e)
  print('Error in getting to collection')

EMBEDDING_SIZE = 384            # 768 | test_embedding_size 384
host = os.getenv("PGHOST")
database = os.getenv("PGDATABASE")
usr = os.getenv("PGUSER")
password = os.getenv("PGPASSWORD")

sql = pg8000.native.Connection(user=usr, password=password, host=host, database=database)

def print_file(txt, filename='printfile_out.txt', typ='w'):
  if not filename: filename='printfile_out.txt'
  filename = 'logs/' + filename

  if isinstance(txt, list):
    with open(filename, 'w'):
      print()
    with open(filename, 'a') as file:
      for item in txt:
        file.write('\n ===')
        file.write(str(item))
      print('Logged in ', filename)
      return

  with open(filename, typ) as file:
    file.write('\n\n ===================================================')
    file.write(str(txt))
  print('Logged in ', filename)

def pg_pullToEmbed():
  query = '''
    select f.id, f.addr, f.cik
    from filings f
    join companies c on f.cik=c.cik
    where chunks is null and c.mcap is not null
    order by c.mcap desc --offset 200
  '''
  return sql.run(query)

def pg_pullTranscriptsByCik(cik):
  query = '''
  select e.callid, e.symbol, max(f.id) as last_filing_id--, last_filing.id as last_filing_id, e.id 
  from earningsCalls e
  join companies c on c.symbol=e.symbol
  join filings f on f.cik=c.cik
  where c.cik= :cik
  group by e.symbol, e.callid
  '''
  return sql.run(query, cik=str(cik))

def pg_pullTranscripts():
  query = '''
    select e.callid, e.symbol, max(f.id) as last_filing_id, c.cik
    from earningsCalls e
    join companies c on c.symbol=e.symbol
    join filings f on f.cik=c.cik
    where e.status = 'new'
    group by e.symbol, e.callid, e.id, c.cik
  '''
  return sql.run(query)


def pg_setCallInsertStatus(callid, status):
  if status not in ['added']:
    raise Exception('pg_setCallInsertStatus: Invalid status: ' + status)
  query = 'UPDATE earningsCalls set status = :status where callid = :callid'
  return sql.run(query, callid=callid, status=status)


def pg_pullTranscriptMessages(callid):
  query = '''
    select txt, agent from earningsMessages where callid = :callid
  '''
  return sql.run(query, callid=str(callid))

def pg_update_chunks_nr(filingID, chunksNr):
  sql_query = "UPDATE filings set chunks = :chunksNr where id = :filingID"
  sql.run(sql_query, filingID=filingID, chunksNr=chunksNr)
  print('pg update done')

def pg_chunks_proc_inprogress(filingID):
  sql_query = "UPDATE filings set chunks = -1, lastmodified=current_timestamp where id = :filingID"
  sql.run(sql_query, filingID=filingID)

def pg_reset_chunks_nr(filingID):
  sql_query = "UPDATE filings set chunks = NULL where id = :filingID"
  sql.run(sql_query, filingID=filingID)

def pg_update_modified(filingID):
  sql_query = "UPDATE filings set lastModified = current_timestamp where id = :filingID"
  sql.run(sql_query, filingID=filingID)

def pg_get_injections():
  sql_query = "SELECT code, names, requirements, embedding_text from injections"
  res = sql.run(sql_query)
  obj = {"financial_terms":{}, "formulas":{}, "embedding_text":{}}
  for item in res:
    code = item[0]
    names_arr = item[1].split(',')
    requirements = item[2]
    embedding_text = item[3]
    obj["financial_terms"][code] = names_arr
    obj["formulas"][code] = requirements
    obj["embedding_text"][code] = embedding_text
  return obj

def pg_write_convo_summary(convoid, summary):
  sql_query = "UPDATE conversations set summary= :summary where convoid= :convoid"
  sql.run(sql_query, convoid=convoid, summary=summary)

def mv_insert_data(entities):
  insert_result = collection.insert(entities)
  collection.flush()
  print(f"Inserted data into '{collection.name}'. Number of entities: {collection.num_entities}")
  return insert_result

def mv_create_collection(name, fields, description):
  schema = CollectionSchema(fields, description)
  collection = Collection(name, schema, consistency_level="Strong")
  return collection

def mv_create_index(collection, field_name, index_type, metric_type, params={}):
  index = {"index_type": index_type, "metric_type": metric_type, "params": params}
  collection.create_index(field_name, index)

def mv_getCollections():
  cols = utility.list_collections()
  print(cols)
  return cols

def mv_select_all():
  # Retrieve the primary key field name
  primary_field = [field.name for field in collection.schema.fields if field.is_primary][0]
  #print(primary_field)
  # Load the collection into memory for search
  collection.load()
  # Query to fetch all IDs (assuming primary field is "id")
  expr = f"{primary_field}"  # Update this query based on your data
  entities = collection.query(expr, output_fields=[primary_field])
  print(entities)

def mv_drop_collection(name):
  utility.drop_collection(name)

def mv_create_nowreports_collection():
  from pymilvus.model.hybrid import BGEM3EmbeddingFunction

  ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
  dense_dim = ef.dim["dense"]

  fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="filingID", dtype=DataType.INT64),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR,
                dim=dense_dim)
  ]
  collection = mv_create_collection("nowreports_bge", fields, "testing collection")
  print('Collection created: ', collection)
  return collection

def mv_create_test_collection():
  from pymilvus.model.hybrid import BGEM3EmbeddingFunction

  ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
  dense_dim = ef.dim["dense"]

  fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="filingID", dtype=DataType.INT64),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR,
                dim=dense_dim),
    FieldSchema(name="isTranscript", dtype=DataType.BOOL)
  ]

  collection = mv_create_collection("test", fields, "testing collection")
  print('Collection created: ', collection)
  return collection

def mv_create_beta_collection():
  from pymilvus.model.hybrid import BGEM3EmbeddingFunction

  ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
  dense_dim = ef.dim["dense"]

  fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="filingID", dtype=DataType.INT64),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR,
                dim=dense_dim),
    FieldSchema(name="isTranscript", dtype=DataType.BOOL)
  ]

  collection = mv_create_collection("nowreports_beta", fields, "beta collection")
  print('Collection created: ', collection)
  return collection

def mv_check_filingID(filingID):
  dir = f'test_data/test_vector_{EMBEDDING_SIZE}'
  if('--wdir' in sys.argv): dir = '/home/alexandru/Desktop/nowreports_ai/' + dir

  array_from_file = np.loadtxt(f'test_data/test_vector_{EMBEDDING_SIZE}')
  res = mv_search_and_query([array_from_file], expr="filingID == " + str(filingID), limit=9999)[0]['dense']
  print('----------', res)
  return res

def mv_get_row_by_filingid(filingid):
  query = 'filingID == ' + str(filingid)
  return collection.query(query, output_fields=["pk"])

def mv_query_by_filingid(filingid, output_fields=["source"], save_embedding=False):
  collection.load()
  query = 'filingID == ' + str(filingid)
  result = collection.query(query, output_fields=output_fields)

  if 'source' in output_fields:
    with open('logs/queryresults.txt', 'w') as file:
      for obj in result:
        file.write('\n===========================================================')
        file.write(str(obj["source"]))
  else: print('result: ', result)

  if len(result) == 0:
    print('mv_query_by_filingid error: No embedding found for id')
    return
  if save_embedding:
    np.savetxt('test_data/test_vector_' + str(EMBEDDING_SIZE), result[0]['sparse_vector'])

  print('Success! Output printed to queryresults.txt')
  return result

#TODO: change
MV_DEF_SEARCH_PARAMS = {"metric_type": "COSINE","params": {"nprobe": 1024, "nlist":1024},
        # search for vectors with a distance smaller than RADIUS
        # "radius": 0.4,
        # # filter out vectors with a distance smaller than or equal to RANGE_FILTER
        # "range_filter" : 0.32
}

def mv_search_and_query(search_vectors, search_params=MV_DEF_SEARCH_PARAMS, expr='', limit=13):
    collection.load()

    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest(search_vectors["sparse"],
                                  "sparse_vector", sparse_search_params, limit=limit, expr=expr)
    dense_search_params = {"metric_type": "L2"}
    dense_req = AnnSearchRequest(search_vectors["dense"],
                                 "dense_vector", dense_search_params, limit=limit, expr=expr)

    # Search topK docs based on dense and sparse vectors and rerank with RRF.
    res = collection.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(k=5),
                            limit=limit, output_fields=['source'])
    #result = collection.search(search_vectors, "embeddings", search_params, limit=limit, output_fields=["source", "filingID"], expr=expr)
    return res

def mv_pg_crosscheck_chunks(filingID, chunksNr=999):
  sql_query = 'SELECT chunks from filings where id = :filingID'
  sql_chunks = sql.run(sql_query, filingID=filingID)[0][0]
  mv_chunks = len(mv_check_filingID(filingID))
  res = {"inp":chunksNr, "sql_chunks":sql_chunks, "mv_chunks":mv_chunks, "ok":False}
  if sql_chunks == mv_chunks == chunksNr: res["ok"] = True
  return res

def mv_get_max_id(filingID):
  result = collection.search(expr='filingID == '+str(filingID))
  print(len(result))


################################################## LEVEL 2 #################################################

def mv_reset_collection():
  mv_drop_collection('nowreports_bge')
  collection = mv_create_nowreports_collection()
  mv_create_index(collection, "sparse_vector", "SPARSE_INVERTED_INDEX", "IP")
  mv_create_index(collection, "dense_vector", "FLAT", "L2")

def mv_delete_filingid(filingid):
  ids = mv_get_row_by_filingid(filingid)
  ids_to_delete = str([str(item["pk"]) for item in ids])
  expr = "pk in " + ids_to_delete
  collection.delete(expr)
  collection.flush()
  print('deleted')

def mv_delete_transcript_embeddings(filingid):
  ids = mv_get_row_by_filingid(filingid)
  ids_to_delete = str([str(item["pk"]) for item in ids])
  expr = "pk in " + ids_to_delete + ' and isTranscript == True'
  collection.delete(expr)
  collection.flush()
  print('deleted')

def batch_del_filingids(filing_ids):
  for filing_id in filing_ids:
    try:
      mv_delete_filingid(filing_id)
    except:
      print()
  print('batch_delete done')

def mv_reset_test_collection():
  mv_drop_collection('test')
  mv_create_test_collection()
  mv_create_index(collection, "sparse_vector", "SPARSE_INVERTED_INDEX", "IP")
  mv_create_index(collection, "dense_vector", "FLAT", "L2")
  collection.load()

def reset_beta_collection():
  mv_drop_collection('nowreports_beta')
  mv_create_beta_collection()
  mv_create_index(collection, "sparse_vector", "SPARSE_INVERTED_INDEX", "IP")
  mv_create_index(collection, "dense_vector", "FLAT", "L2")
  collection.load()

#mv_query_by_filingid(4654, ["source"]) # outputs to file all mv sources
#ids_to_delete = ['1']  # Rep lace with actual IDs of your vectors
#print(mv_pg_crosscheck_chunks(1408,147))
#print(mv_delete_filingid(4))
#print(mv_pg_crosscheck_chunks(4))
#print(mv_check_filingID(4)) # loads random question and prints response
#print(mv_pg_crosscheck_chunks(1528,99))
#mv_reset_collection()
#batch_del_filingids([1,2,3,4,5,6])
#mv_reset_collection()
#reset_beta_collection()