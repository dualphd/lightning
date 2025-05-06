
##  How to evaluate the IQ metrics 
`evaluate.py` - is to get the IQ metrics between the submitted bayer and the gt bayer
```commandline
python ./evaluate.py <input_dir> <output_dir>
```
| Path                                                                                                 | Format | Description        | 
|:-----------------------------------------------------------------------------------------------------|-------:|:-------------------| 
| &#9500;&#9472;&nbsp; input                                                                           |              
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&nbsp; res                                         |   .png | submitted result    |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&nbsp; ref                                         |        |                    |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&nbsp; gt     |   .png | ground truth result |


1. The submitted results in `/res` should be bayers of 10 bits in .bin format 
2. The ground truth in `/ref/gt` should be bayers of 10 bits in .bin format 

