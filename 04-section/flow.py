from metaflow import FlowSpec, step, current, schedule, kubernetes
from rag_tools.filetypes.markdown import Mixin as MarkdownMixin

@schedule(weekly=True)
class MarkdownChunker(FlowSpec, MarkdownMixin):

    @kubernetes(image="registry.hub.docker.com/eddieob/rag:markdown-chunker-mf-task")
    @step
    def start(self):
        """
        Start the flow.
        Try to download the content from the repository.
        """
        self.repo_params = [
            {
                "deployment_url": "docs.metaflow.org",
                "repository_path": "https://github.com/Netflix/metaflow-docs",
                "repository_ref": "master",
                "base_search_path": "docs",
                "exclude_paths": ["docs/v"],
                "exclude_files": ["README.md", "README"],
            }
        ]
        self.df = self.load_df_from_repo_list()
        self.next(self.end)

    @kubernetes(image="registry.hub.docker.com/eddieob/rag:markdown-chunker-mf-task")
    @step
    def end(self):
        print("The flow has ended, with a dataframe of shape: {}".format(self.df.shape))
        print(
            f"""
            You can now use the dataframe to do whatever you want.
            To load it in a notebook, you can use the following code:

                from metaflow import Flow, namespace
                namespace('{current.namespace}')
                run = Run('{current.flow_name}/{current.run_id}')
                df = run.data.df
                print(df.shape)
        """)


if __name__ == "__main__":
    MarkdownChunker()