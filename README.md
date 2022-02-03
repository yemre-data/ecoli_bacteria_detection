# Escherichia coli Bacteria Detection with Deep Learning(SSD Model) on Mother Machine



![Screenshot from 2022-02-03 15-29-21](https://user-images.githubusercontent.com/64874122/152441180-34b580d2-3259-4273-979b-3cda840a2727.png)



## Project Description 

To overcome the constraints of microcolony experiments, microfluidic devices are specifically designed to spatially isolate and align single cells or lineages of cells. These devices contain geometrically fitting microstructures to restrict cells to spatially regular patterns. This geometrical restriction greatly facilitates the task of image analysis.
It is clear that segmentation accuracy directly affects cell tracking. It therefore makes sense to approach the design of the detection method from the point of view of drawing the bounding boxes inside dead-end channels: The microfluidic device is simpler than the 2D microcolonies, with key information already provided because the cells are restricted to grow in a vertical channel. This type of structure can help to determine what methods of detection are truly useful compared to those used to segment 2D microcolonies. The successful detection allows later an accurate segmentation by U-Net at pixel level.
Single shot multi-box(SSD) detector is a very fast and high-accuracy object detection algorithm. SSD has no delegated region proposal network and predicts the boundary boxes and the classes directly from feature maps in one single pass. SSD reaches an advanced detection level and satisfies our demands.


## Aims of the Project

This project aims to create a SSD-based framework. We will construct an adapted model based on the previous generated and preprocessed datasets. Transformation of the bounding-box format ROI (Region of interest) into coco-dataset format is needed in the first place. The modeling with the specific ratio of the bounding box will improve the accuracy. Fine-tuning the parameters is also the core work to finish this project. A developer-friendly interface is appreciated for the future project of tracking and lineage analysis.

