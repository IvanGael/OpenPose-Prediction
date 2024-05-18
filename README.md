### Person's body parts or joint position detection using openCV

- Visualization 1 : Without body parts annotation

![Demo1](demos/demo1.png)


- Visualization 2 : With body parts annotation

![Demo2](demos/demo2.png)


- Visualization 3 : Body part confidence score

![Demo3](demos/demo3.png)


- Visualization 4 : Average confidence score and number of detected body parts

![Demo4](demos/demo4.png)


Example usage
```
py main.py visualization1 --input models/model6.jpg --model graph_opt.pb --width 368 --height 368 --thr 0.2
```