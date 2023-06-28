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
        assert len(batches) == 1
        assert len(batches[0]) == 2
