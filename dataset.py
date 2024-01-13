from blender_ray import *
import pickle

class Dataset():
    def __init__(self, root_dir, img_wh):
        super(Dataset, self).__init__()
        self.root_dir = root_dir
        self.img_wh = img_wh
        self.prepare_data()

    def prepare_data(self):
        kwargs = {'root_dir':  self.root_dir,
                  'img_wh': tuple(self.img_wh)}
        self.train_dataset = BlenderDataset(split='train', **kwargs)
        print(f"train bingo!")
        #self.test_dataset = BlenderDataset(split='val', **kwargs)
        #print(f"test bingo!")


if __name__ == '__main__':
    root_dir = 'data/smallcity'
    img_wh = (1920, 1080)
    dataset_instance = Dataset(root_dir, img_wh)

    #save pkl file
    file_path = 'data/smallcity/training_data.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(dataset_instance.train_dataset.all_rays.numpy(), f)

    #file_path = 'data/smallcity/testing_data.pkl'
    #with open(file_path, 'wb') as f:
    #    pickle.dump(dataset_instance.test_dataset.all_rays.numpy(), f)

    #print(f"Rays data saved to the file: {root_dir}") 