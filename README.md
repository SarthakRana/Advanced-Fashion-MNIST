# 👗 Advanced Fashion MNIST 👖

View the project demo here : https://bit.ly/34pCZyL<br>

![](https://imagesvc.meredithcorp.io/v3/mm/image?q=85&c=sc&poi=face&w=2000&h=1047&url=https%3A%2F%2Fstatic.onecms.io%2Fwp-content%2Fuploads%2Fsites%2F23%2F2020%2F06%2F26%2Fclothing-2000.jpg)

This is an advanced version of the popular CNN based project - Fashion MNIST. In this project, a CNN based model is trained on colored image dataset and is capable of detecting 
not only the class of the cloth but also its color correctly. I have achieved this using the Functional API in keras where I have trained 2 sub-networks - one for category 
and one for color. 

## This project has been completed in 3 parts :

Part 1 : Best suited Model Architecture Experimentation - [check here](https://nbviewer.jupyter.org/github/SarthakRana/Advanced-Fashion-MNIST/blob/main/Best_CNN_Arch_Experiment.ipynb)<br>
Part 2 : Structuring the project into .py files<br>
Part 3 : Making the Web app.

## 1. Prerequisites

You need to have the following dependecies before running the project:

- pandas `pip install pandas`
- numpy `pip install numpy`
- scipy `pip install scipy`
- scikit learn `pip install scikit-learn`
- streamlit `pip install streamlit`
- matplotlib `pip install matplotlib`
- seaborn `pip install seaborn`
- cv2 `pip install opencv-python`
- PIL `pip install Pillow`
- Tensorflow `pip install tensorflow`

**NOTE** : *You can get the dataset from [here](https://drive.google.com/drive/folders/1Mj1LrO5cOm9YX19qztY5-JZYd1z5soXP?usp=sharing)*

**NOTE** : *I used Google Colab for the experimentation part and selecting the final model because of the GPU provided in Colab notebooks.*

## 2. Installing

Download the project on your local system with one of the following ways:

1. You can clone the repo using the Github CLI:
```
gh repo clone SarthakRana/Advanced-Fashion-MNIST
```
2. Download the ZIP folder for this project and extract in your local working directory.

## 3. Usage

### 3.1 Web App

1. Install all dependencies mentioned in __Prerequisites__.
2. Open CLI/prompt and make sure Streamlit is installed by running the command `streamlit --version`. You should see something like this : `Streamlit, version 0.67.1`.
3. Do this for all other dependencies as well just to make sure everything is in right place and you are good to go.
4. Go to your working directory(where you have placed the .py file and other components) and open CLI/prompt there.
5. Type in the following command and press Enter :<br>
   ```streamlit run app.py```<br>
   Please wait for 5-10 seconds for command to run.
6. A browser widow should open up with the app running.
7. Enjoy :)

### 3.2 Project(.py / .ipynb)

1. Install all dependencies mentioned in __Prerequisites__.
2. Place the contents of project folder in your working directory.
3. If you prefer .ipynb file, simply open Jupyter Notebooks/Jupyter Lab and run the .ipynb files.
4. It you prefer .py files, simply fire up CLI and run main.py using ```python main.py```
4. All project related files like models, dataset, test images and encoders will be saved in the same directory as you run the files.

## 4. Project Screenshots

Below are some screenshots from the web app.

![](https://github.com/SarthakRana/Advanced-Fashion-MNIST/blob/main/Screenshots/Screenshot%20(18).png)
![](https://github.com/SarthakRana/Advanced-Fashion-MNIST/blob/main/Screenshots/Screenshot%20(19).png)
![](https://github.com/SarthakRana/Advanced-Fashion-MNIST/blob/main/Screenshots/Screenshot%20(20).png)
![](https://github.com/SarthakRana/Advanced-Fashion-MNIST/blob/main/Screenshots/Screenshot%20(21).png)
![](https://github.com/SarthakRana/Advanced-Fashion-MNIST/blob/main/Screenshots/Screenshot%20(22).png)
![](https://github.com/SarthakRana/Advanced-Fashion-MNIST/blob/main/Screenshots/Screenshot%20(23).png)
![](https://github.com/SarthakRana/Advanced-Fashion-MNIST/blob/main/Screenshots/Screenshot%20(26).png)
![](https://github.com/SarthakRana/Advanced-Fashion-MNIST/blob/main/Screenshots/Screenshot%20(27).png)
![](https://github.com/SarthakRana/Advanced-Fashion-MNIST/blob/main/Screenshots/Screenshot%20(28).png)

## 5. Roadmap

See the open issues for a list of proposed features (and known issues)(if any).
If your issue is not listed in the already open issues, you can open up a new one.

## 6. To-do List

- [ ] Deploy model on Heroku
- [ ] Try out famous model architectures like AlexNet, VGG16/19, etc.
- [ ]

## 7. Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are greatly appreciated.

  1. Fork the Project.
  2. Create your Feature Branch.
  3. Commit your Changes.
  4. Push to the Branch.
  5. Open a Pull Request.

## 8. Authors

NOTE : Your name will be added here if I merge your pull request.

Sarthak Rana (https://www.linkedin.com/in/sarthakrana/)
