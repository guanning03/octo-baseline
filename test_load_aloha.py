from octo.data.gsm.aloha_mobile import load_dataset
from octo.data.utils.format import pytree_display

dataset_path = '/data1/zhuxiaopei/shrimp_tfrecords'

if __name__ == '__main__':
    dataset = load_dataset(dataset_path)
    print(type(dataset))
    print('successfully loaded dataset')
    for step in dataset.take(1):
        print(step['steps'])