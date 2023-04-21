from torchvision.datasets import CIFAR10
class artCIFAR10(CIFAR10):
    """artCIFAR10
    """

    base_folder = "artcifar-10-batches-py"
    url = "https://artcifar.s3.us-east-2.amazonaws.com/artcifar-10-python.tar.gz"
    filename = "artcifar-10-python.tar.gz"
    tgz_md5 = "a6b71c6e0e3435d34e17896cc83ae1c1"
    train_list = [
        ["data_batch_1", "866ea0e474da9d5a033cfdb6adf4a631"],
        ["data_batch_2", "bba43dc11082b32fa8119cba55bebddd"],
        ["data_batch_3", "4978cbeb187b89e77e31fe39449c95ec"],
        ["data_batch_4", "1de1d0514f0c9cd28c5991386fa45f12"],
        ["data_batch_5", "81a7899dd79469824b75550bc0be3267"],
    ]

    test_list = [
        ["test_batch", "2ce6fb1ee71ffba9cef5df551fcce572"],
    ]
    meta = {
        "filename": "meta",
        "key": "styles",
        "md5": "6993b54be992f459837552f042419cdf",
    }