from metaflow import FlowSpec, step, Flow, namespace, current, kubernetes, environment, trigger_on_finish
import os

env_vars = {
    'PINECONE_API_KEY': ...,  
    'GCP_ENVIRONMENT': "us-central1-gcp",
    "TOKENIZERS_PARALLELISM": "false"
}

@trigger_on_finish(flow='DataTableProcessor')
class PineconeVectorIndexer(FlowSpec):

    index_name = "metaflow-documentation"
    embedding_model = "paraphrase-MiniLM-L6-v2"
    embedding_target_col_name = "contents"

    def find_processed_df(self):
        namespace(None)
        try:
            run = current.trigger.run
        except AttributeError as e:
            run = Flow('DataTableProcessor').latest_successful_run
        return run.data.processed_df

    @kubernetes(image="registry.hub.docker.com/eddieob/rag:all")
    @step
    def start(self):
        self.next(self.create_index)

    @kubernetes(image="registry.hub.docker.com/eddieob/rag:all")
    @environment(vars=env_vars)
    @step
    def create_index(self):

        from rag_tools.databases.vector_database import PineconeDB
        from rag_tools.embedders.embedder import SentenceTransformerEmbedder
        import pandas as pd

        # fetch data and embed it
        self.data = self.find_processed_df()
        encoder = SentenceTransformerEmbedder(self.embedding_model, device="cpu")
        docs = self.data[self.embedding_target_col_name].tolist()
        self.ids = list(range(1, len(docs) + 1))
        embeddings = encoder.embed(docs)
        self.dimension = len(embeddings[0])

        # create the index
        db = PineconeDB()
        db.create_index(self.index_name, dimension=self.dimension)

        # put the vectors in the index - idempotent
        db.upsert(self.index_name, embeddings, docs, self.ids)

        self.next(self.end) 

    @kubernetes(image="registry.hub.docker.com/eddieob/rag:all")
    @environment(vars=env_vars)
    @step
    def end(self):

        from rag_tools.databases.vector_database import PineconeDB
        from rag_tools.embedders.embedder import SentenceTransformerEmbedder
        import pinecone

        # create_index is idempotent
        db = PineconeDB()

        # search the index in a test query
        K = 3
        test_prompt = "aws"
        encoder = SentenceTransformerEmbedder(self.embedding_model, device="cpu")
        self._test_search_vector = encoder.embed([test_prompt])[0]
        self._test_results = db.vector_search(self.index_name, self._test_search_vector, k=K).to_dict()

        for result in self._test_results['matches']:
            print("\n\nid: {} - score: {} \n\n{}\n\n".format(result['id'], result['score'], result['metadata']['text']))
            print("===============================================")

        print("\n\n Flow is done, check for results in the {} index at https://app.pinecone.io/.".format(self.index_name))


if __name__ == '__main__':
    PineconeVectorIndexer()