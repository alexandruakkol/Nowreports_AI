from pymilvus import Collection, connections, utility, FieldSchema, DataType, CollectionSchema
from dotenv import load_dotenv
import pg8000.native
import os
import numpy as np

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
  collection = Collection('nowreports')
except:
  print('Error in getting to collection: nowreports')

host = os.getenv("PGHOST")
database = os.getenv("PGDATABASE")
usr = os.getenv("PGUSER")
password = os.getenv("PGPASSWORD")

EMBEDDING_SIZE = 768

sql = pg8000.native.Connection(user=usr, password=password, host=host, database=database)

def print_file(txt, filename='printfile_out.txt', typ='w'):
  if not filename: filename='printfile_out.txt'
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
    order by c.mcap desc offset 100
  '''
  return sql.run(query)

def pg_update_chunks_nr(filingID, chunksNr):
  sql_query = "UPDATE filings set chunks = :chunksNr where id = :filingID"
  sql.run(sql_query, filingID=filingID, chunksNr=chunksNr)

def pg_chunks_proc_inprogress(filingID):
  sql_query = "UPDATE filings set chunks = -1 where id = :filingID"
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

def mv_insert_data(entities):
  insert_result = collection.insert(entities)
  collection.flush()
  print(f"Inserted data into '{collection.name}'. Number of entities: {collection.num_entities}")
  return insert_result

def mv_create_collection(name, fields, description):
  schema = CollectionSchema(fields, description)
  collection = Collection(name, schema, consistency_level="Strong")
  return collection

def mv_create_index(collection, field_name, index_type, metric_type, params):
  index = {"index_type": index_type, "metric_type": metric_type, "params": params}
  collection.create_index(field_name, index)

def mv_getCollections():
  return utility.list_collections()

def mv_select_all():
  # Retrieve the primary key field name
  primary_field = [field.name for field in collection.schema.fields if field.is_primary][0]
  print(primary_field)
  # Load the collection into memory for search
  collection.load()

  # Query to fetch all IDs (assuming primary field is "id")
  expr = f"{primary_field}"  # Update this query based on your data
  entities = collection.query(expr, output_fields=[primary_field])
  print(entities)

def mv_drop_nowreports_collection():
    utility.drop_collection('nowreports')

def mv_create_nowreports_collection():
  fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="filingID", dtype=DataType.INT64),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_SIZE)
  ]
  collection = mv_create_collection("nowreports", fields, "nowreports collection")
  print('Collection created: ', collection)
  return collection

def mv_check_filingID(filingID):
  array_from_file = np.loadtxt(f'test_vector_{EMBEDDING_SIZE}')
  res = mv_search_and_query([array_from_file], expr="filingID == " + str(filingID), limit=9999)[0]
  return res
  #print('Found ' + str(len(res))+ ' entries.')

def mv_get_row_by_filingid(filingid):
  query = 'filingID == ' + str(filingid)
  return collection.query(query, output_fields=["pk"])

def mv_query_by_filingid(filingid, output_fields=["source"], save_embedding=False):
  collection.load()
  query = 'filingID == ' + str(filingid)
  result = collection.query(query, output_fields=output_fields)
  with open('queryresults.txt', 'w') as file:
    for obj in result:
      file.write('\n===========================================================')
      file.write(str(obj["source"]))
  if len(result) == 0:
    print('mv_query_by_filingid error: No embedding found for id')
    return
  if save_embedding:
    np.savetxt('test_vector_' + str(EMBEDDING_SIZE), result[0]["embeddings"])

  print('Success! Output printed to queryresults.txt')
  return result

MV_DEF_SEARCH_PARAMS = {"metric_type": "L2","params": {"nprobe": 32},
        # search for vectors with a distance smaller than RADIUS
        "radius": 0.4,
        # filter out vectors with a distance smaller than or equal to RANGE_FILTER
        "range_filter" : 0.32
}

def mv_search_and_query(search_vectors, search_params=MV_DEF_SEARCH_PARAMS, expr='', limit=13):
    print('-----limit', limit)
    collection.load()
    result = collection.search(search_vectors, "embeddings", search_params, limit=limit, output_fields=["source", "filingID"], expr=expr)
    return result

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
  mv_drop_nowreports_collection()
  collection = mv_create_nowreports_collection()
  mv_create_index(collection, "embeddings", "IVF_FLAT", "L2", {"nlist": 128})

def mv_delete_filingid(filingid):
  ids = mv_get_row_by_filingid(filingid)
  ids_to_delete = str([str(item["pk"]) for item in ids])
  expr = "pk in " + ids_to_delete
  collection.delete(expr)
  collection.flush()

def batch_del_filingids(filing_ids):
  for filing_id in filing_ids:
    mv_delete_filingid(filing_id)
  print('batch_delete done')

#ids_to_delete = ['1']  # Rep lace with actual IDs of your vectors

#mv_query_by_filingid(53, ["source"]) # outputs to file all mv sources
#print(mv_pg_crosscheck_chunks(1408,147))
#print(mv_delete_filingid(4))
#print(mv_pg_crosscheck_chunks(4))
#print(mv_check_filingID(4)) # loads random question and prints response
#print(mv_pg_crosscheck_chunks(1528,99))
#mv_reset_collection()
#batch_del_filingids([1528,5,6,7,8,9,10,11,12,33,34,35,36,13,14,15,16,17,18,19,20,21,22,23,24])

