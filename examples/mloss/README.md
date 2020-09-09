# Results

| Algorithm | Search Time | Final Accuracy | Final #Params | Final FLOPS | CheckPoint & Log (Drive) |
| --------- | ----------- | -------------- | ------------- | ----------- | -----------------------  |
| ENAS      | 06h 16m 54s | 97.30%         | 4.20M         | 1303M       |[OneDrive](https://1drv.ms/u/s!AoimKNYWyNYsgeloYQwDj1iSPrYTeA?e=yuo8Vw)|
| DARTS     | 09h 04m 35s | 97.10%         | 2.81M         | 892M        |[OneDrive](https://1drv.ms/u/s!AndYiSSFXw7ijgRd8SeYnjzinRZl?e=lJh2ts)  |
| SNAS      | 08h 03m 21s | 97.02%         | 3.18M         | 1029M       |[OneDrive](https://1drv.ms/u/s!AndYiSSFXw7ijhC2jwC4B7dSwyDC?e=QWpthl)  |
| PC-DARTS  | 02h 57m 03s | 97.43%         | 4.26M         | 1343M       |[OneDrive](https://1drv.ms/u/s!AoimKNYWyNYsgelsu8tfQwZ748CGeA?e=4hs8ai)|


## Notes:

* All search and final are done with 1 NVIDIA 2080Ti.
* ENAS Peak GPU memory: **8.2GiB** during search (batch size 256), **9.2GiB** during final training (batch size 64).
* PC-DARTS peak GPU memory: **9.5GiB** during search (batch size=192, k=4), **8.9GiB** during final training (batch size 64).
* Learning rates have been adjusted in accordance with the change of batch sizes, due to hardware limitation.
* Time cost may not be accurate due to interference from other projects running on the same server.
