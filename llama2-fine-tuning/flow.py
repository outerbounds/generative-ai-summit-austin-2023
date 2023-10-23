from metaflow import FlowSpec, step, pypi, kubernetes, environment, S3, current

N_GPU = 8
N_NODES = 1


def make_tar_bytes(source_dir):
    from tarfile import ExtractError, TarFile
    from io import BytesIO

    buf = BytesIO()
    with TarFile(mode="w", fileobj=buf) as tar:
        tar.add(source_dir)
    return buf.getvalue()


class CoreweaveFineTuneWithDolly15K(FlowSpec):
    @step
    def start(self):
        self.next(self.train)

    @environment(
        vars={
            "CUDA_VISIBLE_DEVICES": ",".join([str(i) for i in list(range(N_GPU))]),
        }
    )
    @pypi(
        python="3.10.10",
        packages={
            "transformers": "4.31.0",
            "peft": "0.4.0",
            "datasets": "2.14.5",
            "bitsandbytes": "0.40.2",
            "accelerate": "0.23.0",
            "trl": "0.4.7",
            "scipy": "1.11.3",
            "tensorboard": "2.14.1",
        },
    )
    @kubernetes(gpu=N_GPU, cpu=32, memory=64000)
    @step
    def train(self):
        # fine tune llama
        from model import main
        main()

        # zip and push huggingface model and tokenizer
        from params import dst_model_name

        self.out_path = dst_model_name + "_" + current.run_id + ".tar"
        with S3(run=self) as s3:
            s3.put(key=self.out_path, obj=make_tar_bytes(dst_model_name))

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    CoreweaveFineTuneWithDolly15K()
