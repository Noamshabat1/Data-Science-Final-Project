from data_processing_runner import main as data_processing_main
from model.build_data import load_and_merge_data
from model.models import main as model_main
from community_detection.community_detection import main as community_detection_main


def main():
    print(" >>> Starting date preprocessing...")
    data_processing_main()

    print(" >>> Starting Community Detection...")
    community_detection_main()

    print(" >>> Starting model data building...")
    df_final = load_and_merge_data()
    print("Sample of the final dataset:")
    print(df_final.head())

    print(" >>> Starting model training and evaluation...")
    model_main()


if __name__ == "__main__":
    main()
