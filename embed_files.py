from datetime import datetime
from uuid import uuid4
from qdrant_client.models import PointStruct
from memory_profiler import profile

#Takes Chunked Text and embeds files with openai embedding
def EmbedFiles(client, chunk, tenantid, propertymanagerid, propertyid,unitid, upload_session_id, pagenumber, sourcedocname, chunkindex, company_id):
    #If creating a tenant, sets tenant id null
    if tenantid is None:
        tenantid= ""
    #Uses openAI embeding to embed chunks
    try:
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-large"
        )
        #Gets embedded vector from openai
        vector = response.data[0].embedding
        #prepares payload for vector db
    except Exception as e:
        print("Error Embedding Files", e)
    return PointStruct(
        id=str(uuid4()),
        vector = vector,
        payload={
            "tenantid": tenantid,
            "propertymanagerid": propertymanagerid,
            "propertyid": propertyid,
            "unitid": unitid,
            "pageNumber": pagenumber,
            "source_doc": sourcedocname,
            "upload_date": datetime.utcnow().isoformat(),
            "text": chunk,
            "session_id" : upload_session_id,
            "source_id" : f"{upload_session_id}_{tenantid}_{sourcedocname}_{pagenumber}_{chunkindex}",
            "managementcompany_id" : company_id,
            "highlight_id" : str(uuid4())
        })
   
    