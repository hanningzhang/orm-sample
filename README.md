# orm-sample

go to `\upload`

generate data:
```
python generate_numina_deepseek.py --start 0 --end 50000 --output_dir xxx.json
```
There are about 850K math problems in the Numina dataset, you may use `--start` and `--end` to specify the part you wish to generate.
It will take a lot of time to generate all the data at one time.
So it is better to use multiple process to generate by part (using --start --end).
