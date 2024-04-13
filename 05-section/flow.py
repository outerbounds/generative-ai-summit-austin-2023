from metaflow import FlowSpec, step, S3, Parameter, kubernetes, card, current, Flow, trigger_on_finish, namespace
from metaflow.cards import Image, Markdown
from matplotlib import pyplot as plt

try: # packages included in task Docker container
    import tldextract
    import seaborn as sns
    sns.set_style("dark")
    COLORS = {
        "purple": "#98A1E1",
        "light-purple": "#DADEFB",
        "gold": "#F0C054"
    }
except ImportError:
    print("Skipping seaborn import...")


@trigger_on_finish(flow='MarkdownChunker')
class DataTableProcessor(FlowSpec):

    save_processed_df = Parameter(
        "save_processed_df",
        help="Whether to save the processed dataframe to the run. Useful for local runs.",
        default=True,
        type=bool,
    )

    parent_flow = Parameter(
        "parent_flow",
        help="The flow id of the parent flow to process.",
        default='MarkdownChunker',
        type=str,
    )

    n_bins = Parameter(
        "n_bins",
        help="The number of bins to use in the histogram.",
        default=100,
        type=int,
    )

    word_count_threshold = Parameter(
        "word_count_threshold",
        help="The word count threshold to use in the histogram.",
        default=10,
        type=int,
    )

    char_count_threshold = Parameter(
        "char_count_threshold",
        help="The char count threshold to use in processing.",
        default=25,
        type=int,
    )

    def plot_char_word_histogram(self, char_count_threshold=0, word_count_threshold=0, _df=None, title="", ):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0] = _df.char_count.plot.hist(bins=self.n_bins, color=COLORS['purple'], ax=ax[0])
        ax[1] = _df.word_count.plot.hist(bins=self.n_bins, color=COLORS['light-purple'], ax=ax[1])
        if char_count_threshold > 0 or word_count_threshold > 0:
            ax[0].set_xlabel("Filtered character count > %d" % char_count_threshold)
            ax[1].set_xlabel("Filtered word count > %d" % word_count_threshold)
        ax[0].set_xscale("log")
        ax[1].set_xscale("log")
        ax[0].set_ylabel("Frequency")
        ax[0].spines.right.set_visible(False)
        ax[0].spines.top.set_visible(False)
        ax[1].spines.right.set_visible(False)
        ax[1].spines.top.set_visible(False)
        fig.suptitle(title, fontsize=24)
        fig.tight_layout()
        assert fig is not None, "Figure is None, check plot_char_word_histogram."
        return fig

    def plot_tld_count(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        self.processed_df.groupby('tld').count()['index'].sort_values(ascending=False).plot.bar(
            ax=ax, color=COLORS['gold']
        )
        fig.suptitle("Top-level domain representation in the dataset", fontsize=24)
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        assert fig is not None, "Figure is None, check plot_tld_count."
        return fig

    @kubernetes(image="registry.hub.docker.com/eddieob/rag:markdown-chunker-mf-task")
    @card
    @step
    def start(self):

        try:
            run = current.trigger.run
        except AttributeError as e:
            namespace(None)
            parent_flow = Flow(self.parent_flow)
            run = parent_flow.latest_run 
        
        if not run.successful:
            print("Skipping processing of unsuccessful run {}.".format(run.id)) 
            self.processed_df = None

        else:

            current.card.append(Markdown(f"""# Processing data table from run {run.id}"""))

            current.card.append(Markdown(f"""## Filtering rows"""))
            df = run.data.df
            fig = self.plot_char_word_histogram(_df = df, title="Before filtering")
            current.card.append(Image.from_matplotlib(fig))

            # Filter out rows with less than N words.
            _df = df[df.word_count > self.word_count_threshold]

            # Filter out rows with less than M chars.
            _df = _df[_df.char_count > self.char_count_threshold]

            # Feature: Add a column for the top level domain.
            _df['tld'] = _df['page_url'].apply(lambda url: "https://" + tldextract.extract(url).fqdn)

            # Reindex and keep index in upstream dataframe.
            _df.reset_index(inplace=True)
            _df.index = range(len(_df))

            fig = self.plot_char_word_histogram(
                word_count_threshold=self.word_count_threshold, 
                char_count_threshold=self.char_count_threshold,
                _df = _df, title="After filtering"
            )
            current.card.append(Image.from_matplotlib(fig))

            ### ADD MORE FILTERS HERE.
            print("Filtered dataframe from shape {} to shape {}.".format(
                df.shape, _df.shape))
            self.processed_df = _df

            # Plot the number of rows per TLD.
            fig = self.plot_tld_count()
            current.card.append(Image.from_matplotlib(fig))

            ### ADD MORE SUMMARY STATS HERE.

        self.next(self.end)

    @kubernetes(image="registry.hub.docker.com/eddieob/rag:markdown-chunker-mf-task")
    @step
    def end(self):

        import os

        print("The {} run {} has ended, with a dataframe of shape: {}".format(
            current.flow_name, current.run_id,
            self.processed_df.shape))
        print(
            f"""
            You can now use the dataframe to do whatever you want.
            To load it in a notebook, you can use the following code:

                from metaflow import Flow, namespace
                namespace('{current.namespace}')
                run = Run('{current.flow_name}/{current.run_id}')
                processed_df = run.data.processed_df
                print(processed_df.shape)
        """
        )

        # if self.save_processed_df:

        #     import os
        #     self.is_local = os.environ.get("METAFLOW_RUNTIME_ENVIRONMENT", "local") == "local"

        #     self.processed_df_path = os.path.abspathe(
        #         os.path.join('processed_df_%s.csv' % current.run_id))
        #     print(os.getcwd())
        #     print("Saving processed dataframe to %s" % self.processed_df_path)
        #     self.processed_df.to_csv(self.processed_df_path, index=False)
            
        #     if not self.is_local:
        #         with S3(run=self) as s3:
        #             s3.put_files([
        #                 (self.processed_df_path.split('/')[-1], self.processed_df_path)
        #             ])

if __name__ == '__main__':
    DataTableProcessor()