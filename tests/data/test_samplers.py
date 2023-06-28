from mini_rec_sys.data import Sampler, BatchedSequentialSampler, SessionDataset, Session


class TestSampler:
    def test_batched_sequential_sampler(self, default_session_data, capsys):
        dataset = SessionDataset(
            id_name="session_id",
            store_fn=lambda id, row: Session(
                session_id=id,
                positive_items=row["positive_items"],
                negative_items=row["negative_items"],
                positive_relevances=row["positive_relevances"],
                query=row["query"],
            ),
            data=default_session_data,
        )
        sampler = BatchedSequentialSampler(dataset, batch_size=2, drop_last=False)
        batches = [batch for batch in iter(sampler)]
        N = len(default_session_data)
        assert len(batches) == N / 2
        assert len(batches[0]) == 2
        collect_batches = set()
        for batch in batches:
            collect_batches.update(batch)
        assert collect_batches == set(default_session_data.keys())
