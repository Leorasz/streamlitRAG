from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)


docs = [
    "this is one document",
    "and another document"
]

embeddings = embed_model.embed_documents(docs)

print(f"We have {len(embeddings)} doc embeddings, each with "
      f"a dimensionality of {len(embeddings[0])}.")

import os
import pinecone

# get API key from app.pinecone.io and environment from console
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or '76793265-e2ab-4fcb-bfed-39a64f7b029a',
    environment=os.environ.get('PINECONE_ENVIRONMENT') or 'gcp-starter'
)


import time

index_name = 'llama-2-military-bases'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=len(embeddings[0]),
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)


index = pinecone.Index(index_name)
index.describe_index_stats()


from datasets import load_dataset

data = load_dataset(
    'Leorasz/military_bases',
    split='train'
)


batch_size = 32

for i in range(0, len(data), batch_size):
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]
    ids = [f"{x['OBJECTID']}" for i, x in batch.iterrows()]
    texts = [f"The military base {x['siteName']} is located in {x['stateNameCode']}. Its operational status is {x['siteOperationalStatus']}, and its reporting component is {x['siteReportingComponent']}" for i, x in batch.iterrows()]
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {'siteName': x['siteName'],
         'text': f"The military base {x['siteName']} is located in {x['stateNameCode']}. Its operational status is {x['siteOperationalStatus']}, and its reporting component is {x['siteReportingComponent']}",
         'siteReportingComponent': x['siteReportingComponent']} for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

index.describe_index_stats()








from torch import cuda, bfloat16
import transformers

model_id = 'meta-llama/Llama-2-13b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)


generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)


res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])

"""Now to implement this in LangChain"""

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)


from langchain.vectorstores import Pinecone

text_field = ''  # field in metadata that contains text content

vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)



from langchain.chains import RetrievalQA

rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever()
)