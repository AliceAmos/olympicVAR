# olympicVAR

THE VERY FIRST AUTOMATIC JUDGE EVER BEEN INVENTED, FOR DIVING SPORTS!

This project is testing and investigating main 2 approaches regarding predicting score on a diving video:
  - Predict each frame by a single pre-trained neural network model.
  - Predict each frame by a 3 different pre-trained neural networks on specific frames and not on the whole video.

  * Notice that the networks were trained on a public dataset - (e.g AQA7 - http://rtis.oit.unlv.edu/datasets.html)
After getting a score for each frame - these predictions create new type of dataset, to be trained by a ML model.
A nice feature added was a sommersaults counter, to be more accurate and simulate the judging method in real life. 
The prediction of the model is actually a final score.

You could run on your local computer - pass the path to the video you filmed (on horizontal view only!) to the predict function in main.py file.
Or when running the server, you'll see a UI on localhost:3000 for uploading video to the Coludinary platform and view your score!


Will be a great reference for other researches on that area of Deep learning.
