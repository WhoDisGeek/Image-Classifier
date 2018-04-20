# Image-Classifier


Create Folder Structure

/Project/ \
-- Data/ \
-- -- ndjson/ \
-- -- -- airplane.ndjson \
-- -- -- banana.ndjson \
-- -- CNN-Total/ \
-- NDJ-Image.py \
-- Split-Data.py \
-- CNN.py \
-- Train-CNN.py \
-- Test-CNN.py


cd /Project/ \
python3 NDJ-Image.py /Project/Data \
python2 Split-Data.py /Project/Data \
python3 Train-CNN.py \
python3 Test-CNN.py   --> This is Optional

Use the Models saved in /Project/CNN/   --> The one with '-Inter' is usually the best performing \
If running Train-CNN.py again, delete directory /Project/Data/CNN/ 
