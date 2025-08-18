Dataset raw from "data.csv"
Load "data.csv" as backup

Pipeline Prep:
    Split raw into train, test with ratio=0.8
    Load "test.csv" as t2

Train train with epochs=10
Evaluate model on test
Print summarize(model)