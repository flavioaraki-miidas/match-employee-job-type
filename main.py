from FlagEmbedding import BGEM3FlagModel
from typing import Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import csv
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import os.path

class JapaneseCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """ include Japanese 。、 to splitter """

    def __init__(self, **kwargs: Any):
        separators = ["\n\n", "\n", "。", "、", " ", ""]
        super().__init__(separators=separators, **kwargs)


text_splitter = JapaneseCharacterTextSplitter(chunk_size=512, chunk_overlap=128)

""" model """
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "BAAI/bge-m3" # Huggingfaceから取得する場合
model_kwargs = {"device": device}
encode_kwargs = {"normalize_embeddings": True}  # Cosine Similarity

embedding = HuggingFaceBgeEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

""" vectore store """
local_save_dir_name = 'data'
local_save_file_name = 'faiss_vectorstore.index'
local_save_file_name_suffix = '.faiss'
search_type = "similarity_score_threshold"
search_result_document_count = 10
search_score_threshold = 0.8
search_lambda_mult = 0.8

embeddings_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=search_score_threshold)
vectorstore = None
retriever = None

if os.path.exists(local_save_dir_name+'/'+local_save_file_name+local_save_file_name_suffix):
    """ if index already exists, load from disk """
    vectorstore = FAISS.load_local(folder_path=local_save_dir_name, index_name=local_save_file_name, embeddings=embedding, allow_dangerous_deserialization=True)
else:
    """ create vectorstore from Miidas CSV file """
    miidas_csv_path = "data/miidas_job_description.csv"


    with open(miidas_csv_path, newline='') as miidas_csvfile:

        miidas_reader = csv.reader(miidas_csvfile, delimiter=',')

        for index, miidas_row in enumerate(miidas_reader):
            if index == 0:
                continue
            print(miidas_row[4], miidas_row[5])
            header = miidas_row[5] + '。'
            miidas_row_text = header + miidas_row[6]
            print(miidas_row_text)
            miidas_texts = text_splitter.split_text(miidas_row_text)
            miidas_metadata = {"job_type_small_id":miidas_row[4], "job_type_small_name":miidas_row[5]}
            miidas_metadatas = []
            for i in miidas_texts:
                miidas_metadatas.append(miidas_metadata)
            
            if vectorstore is None:
                """ initialize vectorstore  """
                vectorstore = FAISS.from_texts(texts=miidas_texts, embedding=embedding, metadatas=miidas_metadatas)
            else:
                vectorstore.add_texts(texts=miidas_texts, metadatas=miidas_metadatas)

        """ save vector index on disk  """
        vectorstore.save_local(folder_path=local_save_dir_name, index_name=local_save_file_name)

retriever = vectorstore.as_retriever()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever
)

""" load industries """
industries_csv_path = "data/industries.csv"
industries = {}
with open(industries_csv_path, newline="") as industries_csvfile:
    industries_reader = csv.reader(industries_csvfile, delimiter=",")
    for index, industry_row in enumerate(industries_reader):
        if index == 0:
            continue
        industry_details = {
            "name": industry_row[1],
            "detail": industry_row[2],
            "industry_large": industry_row[3],
            "industry_middle": industry_row[4],
            "industry_small": industry_row[5],
        }
        industries[industry_row[0]] = industry_details


""" loop employee jobs to match with Miidas' jobs  """
employee_jobs_csv_path = "data/employee-jobs.csv"
output_csv_path = "data/matching_result.csv"
with open(output_csv_path, 'w', encoding='UTF8') as output_csvfile:
    output_writer = csv.writer(output_csvfile)
    output_row = ['company_id','section_label','job_type_label',
        'role_rank_id','post_label',
        'score_1','miidas_job_type_small_id_1','miidas_job_type_small_name_1',
        'score_2','miidas_job_type_small_id_2','miidas_job_type_small_name_2',
        'score_3','miidas_job_type_small_id_3','miidas_job_type_small_name_3',
        'score_4','miidas_job_type_small_id_4','miidas_job_type_small_name_4',
        'score_5','miidas_job_type_small_id_5','miidas_job_type_small_name_5',
    ]
    output_writer.writerow(output_row)
    with open(employee_jobs_csv_path, newline='') as employee_csvfile:
        employee_reader = csv.reader(employee_csvfile, delimiter=',')


        for index, employee_row in enumerate(employee_reader):
            if index == 0:
                    continue

            job_type_label = employee_row[2]

            """ CSVに書き出される行 """
            output_row = [
                employee_row[0],
                employee_row[1],
                job_type_label,
                employee_row[3],
                employee_row[4],
            ]


            if len(job_type_label) == 0:
                """ 従業員の職種名がなければデータをコピーするだけ """
                """ なので15個の空白を追加する """
                for i in range(15):
                    output_row.append("")

                output_writer.writerow(output_row)
                continue

            
            industry_details = industries[employee_row[0]]

            """ query """
            full_query = ("私が所属している「" + 
                industry_details["name"] +
                "」の業種名は「" +
                industry_details["industry_large"] + "," +
                industry_details["industry_middle"] + "," +
                industry_details["industry_small"] + "」" +
                "です。私は「" +
                employee_row[1] +
                "」という組織で「" +
                job_type_label +
                "」を担当しております。"
            )


            """ Example using similarity_score_threshold
            full_query = (employee_row[1] + "," + job_type_label + "\n" +
                industry_details["industry_large"] + "," +
                industry_details["industry_middle"] + "," +
                industry_details["industry_small"] + "\n" +
                industry_details["detail"])
                
            print("full query")
            """ 
            query_chunks = text_splitter.split_text(full_query)
            query = query_chunks[0]
            """
            print("")
            print("query")
            print(query)
            """

            search_results = vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=search_result_document_count,
                kwargs={
                    "lambda_mult":search_lambda_mult,
                    "score_threshold":search_score_threshold
                }
            )

            """ Example using retriever
            search_results = compression_retriever.invoke(query)
            print("search_results =")
            print(search_results)
            raise Exception("")
            """

            miidas_ids = []
            for search_result in search_results:
                document = search_result[0]
                score = search_result[1]

                """ check for duplicates """
                if document.metadata['job_type_small_id'] in miidas_ids:
                    continue
                output_row.append(score)
                output_row.append(document.metadata['job_type_small_id'])
                output_row.append(document.metadata['job_type_small_name'])
                miidas_ids.append(document.metadata['job_type_small_id'])
                if len(miidas_ids) == 5:
                    break

            output_writer.writerow(output_row)
