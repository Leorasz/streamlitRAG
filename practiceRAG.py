from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import streamlit as st
from qdrant_client import QdrantClient
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print("device deviced, it's ", device)
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)


client = QdrantClient(host="localhost", port=6333)
from qdrant_client.http import models

index_name = 'llama-2-military-bases'

@st.cache
def create_collec():
    client.create_collection(
        collection_name=index_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
create_collec()

from datasets import load_dataset

dataset = load_dataset(
    'Leorasz/military_bases',
    split='train'
)

data = dataset.to_pandas()
batch_size = 32


@st.cache
def addData():
    for i in range(0,len(data), batch_size):
        i_end = min(len(data), i+batch_size)
        batch = data.iloc[i:i_end]
        idss = list(range(i,i_end))
        texts = [f"The military base {x['siteName']} is located in {x['stateNameCode']}. Its operational status is {x['siteOperationalStatus']}, and its reporting component is {x['siteReportingComponent']}" for i, x in batch.iterrows()]
        embeds = embed_model.embed_documents(texts)
        client.upsert(
            collection_name = index_name,
            points = models.Batch(
                ids=idss,
                vectors = embeds,
                payloads = [
                    {'siteName': x['siteName'],
                    'text': f"The military base {x['siteName']} is located in {x['stateNameCode']}. Its operational status is {x['siteOperationalStatus']}, and its reporting component is {x['siteReportingComponent']}"} for i, x in batch.iterrows()
                ]
            )
        )
addData()



from torch import cuda, bfloat16
import transformers

model_id = 'meta-llama/Llama-2-13b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
if device == 'cpu':
    print("ERROR, ERROR, USING CPU")

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
@st.cache
def createModel():
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        # quantization_config=bnb_config,
        use_auth_token=hf_auth
    )
model = createModel()
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=256,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)


from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)

def RAGresponse(prompt):
    emb = embed_model.embed_documents([prompt])[0]
    texxt = client.search(
        collection_name=index_name,
        query_vector = emb,
        limit = 1
    )
    print(texxt)
    texxt = texxt[0].payload['text']
    inp = f"You have been asked {prompt}. {texxt}. Please answer the question using this information."
    return llm(inp)


user_input = st.text_input("Enter your question:")
print(user_input)
resp = (llm(str(user_input)))
print(resp)
st.write(resp)
