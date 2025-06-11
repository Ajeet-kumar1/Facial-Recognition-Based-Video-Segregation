Description: 
Given: 
● A reference image containing a person's face. 

● A collection of short video clips (10 seconds to 2 minutes each), which may contain 
one or more people. 

The task is to: 
1. Detect and extract faces from each video clip using Computer Vision. 
2. Use facial recognition algorithms to match the extracted faces to the reference 
image. 
3. Classify the videos into two categories: 
○ Videos that contain the reference face. 
○ Videos that do not contain the reference face.

To install and run this use the following commands

```
git clone https://github.com/Ajeet-kumar1/Facial-Recognition-Based-Video-Segregation.git
cd Facial-Recognition-Based-Video-Segregation
```

```
conda create --name facerecog python==3.10
conda activate facerecog
pip install -r requirements.txt
```

After this run 
```
python main.py
```

