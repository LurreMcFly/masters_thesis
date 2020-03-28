# Master's Thesis
__by: Jonas Lundgren__

My master’s thesis, currently in progress. Written as a part of the Master of Science in Engineering Mathematics degree at the Faculty of Engineering at Lund University.  It was written during the spring of 2020 and was done in collaboration with Sentian.ai in Malmo.


Example of anomaly detection system at work
![train](figures/train.gif)

__GIF explanation__
__Top figure__
- __Blue line__: Time-series from Yahoo Webscope benchmark data set.
- __Red x__: True anomalies
- __Green line _(40 data points)___: Input sliding window.
- __Red line _(100 data points)___: When querying label {anomaly, normal} the oldest 20% in ensamble is updated and trained on this red sliding window, i.e. the 100 most recent data points.
- __Green points__: Queries to human expert
- __Yellow points__: Predicted anomalies
__Bottom figure__
- __Purple line__: Anomaly score
- __Red line__: Threshold = 0.5
