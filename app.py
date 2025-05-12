from flask import Flask, render_template, request , session
from qa_pipeline import get_answer
from flask_session import Session
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from flask import redirect, url_for

index_name = "maindb"

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

PCapi = os.getenv('ragdoctor')
os.environ["PINECONE_API_KEY"] = PCapi
pc = Pinecone(api_key=PCapi)

# Loading PC
vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})

# Gemini init
gemapi = os.getenv('GemAPI')
os.environ["GOOGLE_API_KEY"] = gemapi
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, max_tokens=1024)

# Custom prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a highly knowledgeable and trustworthy MBBS Doctor,Dr Sushruta, specializing in  modern medicine. You are provided context from some of the world’s most authoritative clinical resources, including the Oxford Handbook of Clinical Medicine, Harrison’s Principles of Internal Medicine, The Merck Manual, and other standard medical encyclopedias .

When Users ask/ describe their symptoms, ask questions about diseases like "eg: tell me if you have these symptomps" , Tell patient to do related (desease matches to the symptom) tests if necessary "eg: You need to provide CT scan report/ blood test report.. do these tests" . Once you are sure about the desease You give medication (medicine name )

So Follow like this:
- Make sure patient have this particular desease (think these might be the possible deseases) if not sure tell them to provide more symptoms related to the desease , tell to do medical tests if Needed 
- Once sure about the desease, tell that he got this disease (mention name), If not sure tell them to Do x, y, z tests and give the test result,
- give  Prescription like profesional doctor when confirmed about the desease, (provide from the  sources You are already given, If you dont get source text tell "sorry No Context Available")
- Some general recommendations (eg: diet , lifestyle etc ) related to that desease 
DO it Like a Doctor 
Always act like a professional and empathetic doctor. Never invent or hallucinate treatments.If ask question outside the field give rude reply!! Only make suggestions based on the context given to you from trusted medical literature.

⚠️ Be creative in your response, but strictly grounded in the provided medical context. Don't Add `*` or `**` symbol to answer.
Context:
{context}

Question:
{question}

Answer:
"""
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

@app.route('/', methods=['GET', 'POST'])

def index():
        
    answer = None
    question = ""
    if "history" not in session:
        session["history"] =[]

    # if request.method == 'POST':
    #     question = request.form['question']
    #     temp = qa_chain.invoke(question)
    #     result = [doc.page_content for doc in temp]
    #     answer = result['result']
    #     # answer = get_answer(question)
    #     session["history"].append({"question": question, "answer": answer})
    #     session.modified = True
    # # return render_template('index.html', answer=answer, question=question)
    # return render_template("index.html", history=session["history"], answer=answer, question=question)
    if request.method == 'POST':
        question = request.form['question']
        docs = retriever.invoke(question)

        clean_context = [doc.page_content for doc in docs]
        context_string = "\n\n".join(clean_context)
        # formatting
        final_prompt = custom_prompt.format(context = context_string, question = question)

        response= llm.invoke(final_prompt)
        answer= response.content if hasattr(response, "content") else str(response)
        print(response)


        session["history"].append({
            "question": question,
            "answer": answer
        })
        session.modified = True

        # temp = qa_chain.invoke(question)

        # answer = temp['result']  # The generated answer
        # clean_context = [doc.page_content for doc in temp['source_documents']]  # Just the text

        # session["history"].append({
        #     "question": question,
        #     "answer": answer,
        #     "context": clean_context  # optional if you want to store this
        # })
        # session.modified = True

    return render_template("index.html", history=session["history"], answer=answer, question=question)



@app.route('/clear', methods=['GET', 'POST'])
def clear():
    session["history"] = []
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)
