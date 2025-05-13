import os
import json
import datasets

class GeometryReasoningConfig(datasets.BuilderConfig):
    def __init__(self, tasks, modes, data_dir=None, dataset_path=None, **kwargs):
        super(GeometryReasoningConfig, self).__init__(**kwargs)
        self.tasks = tasks
        self.modes = modes
        self.data_dir = data_dir
        self.dataset_path = dataset_path

class GeometryReasoning(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = GeometryReasoningConfig
    BUILDER_CONFIGS = [
        GeometryReasoningConfig(
            name="default", 
            tasks=["reasoning"], 
            modes=["interleaved_reasoning"],
            version=datasets.Version("1.0.0"),
            description="Geometry reasoning dataset with interleaved text and images",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "input_text": datasets.Value("string"),
                "input_img_paths": datasets.features.Sequence(datasets.Value("string")),
                "label_text": datasets.Value("string"),
                "label_img_paths": datasets.features.Sequence(datasets.Value("string")),
                "task": datasets.Value("string"),
                "train_task": datasets.Value("string"),
                "idx": datasets.Value("int32"),
            }),
        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir if self.config.data_dir else "."
        dataset_path = self.config.dataset_path if self.config.dataset_path else None

        # Load data from the specified path
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, "r") as f:
                raw_data = json.load(f)
            
            if isinstance(raw_data, dict) and "train" in raw_data:
                # Data has train split
                train_data = raw_data["train"]
                
                # Create validation and test splits if not present
                if "validation" not in raw_data and "dev" not in raw_data:
                    val_size = max(1, int(len(train_data) * 0.1))  # 10% for validation
                    val_data = train_data[-val_size:]
                    train_data = train_data[:-val_size]
                else:
                    val_data = raw_data.get("validation", raw_data.get("dev", []))
                
                if "test" not in raw_data:
                    test_size = max(1, int(len(train_data) * 0.1))  # 10% for test
                    test_data = train_data[-test_size:]
                    train_data = train_data[:-test_size]
                else:
                    test_data = raw_data.get("test", [])
            else:
                # Data is just a list, split it
                all_data = raw_data if isinstance(raw_data, list) else raw_data["train"]
                total = len(all_data)
                val_size = max(1, int(total * 0.1))
                test_size = max(1, int(total * 0.1))
                train_size = total - val_size - test_size
                
                train_data = all_data[:train_size]
                val_data = all_data[train_size:train_size+val_size]
                test_data = all_data[train_size+val_size:]
        else:
            # Default empty datasets if no path provided
            train_data, val_data, test_data = [], [], []
            
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data": train_data},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data": val_data},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data": test_data},
            ),
        ]

    def _generate_examples(self, data):
        """Generate examples from loaded data."""
        tasks = self.config.tasks
        modes = self.config.modes
        
        for idx, item in enumerate(data):
            # Add default values for items that might be missing
            item_task = item.get("task", "reasoning")
            item_train_task = item.get("train_task", "interleaved_reasoning")
            
            if ((not tasks or item_task in tasks) and 
                (not modes or item_train_task in modes)):
                input_text = item.get("input_text", "")
                label_text = item.get("label_text", "")
                
                input_img_paths = item.get("input_img_paths", [])
                
                label_img_paths = item.get("label_img_paths", [])
            
                example = {
                    "input_text": input_text,
                    "input_img_paths": input_img_paths,
                    "label_text": label_text,
                    "label_img_paths": label_img_paths,
                    "task": item_task,
                    "train_task": item_train_task,
                    "idx": item.get("idx", idx),
                }
                yield idx, example 