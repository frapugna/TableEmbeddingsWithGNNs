from pyunpack import Archive
import bz2
import json
import os
import pickle
import re
import time
from pymongo import MongoClient

#To parse from scratch
def parse_dataset(first, last):

    dataset_path = "datasets/tobias_tables"

    zip_folder_path = dataset_path + "/original_files"

    unzip_folder_path = zip_folder_path + "/output-table"

    zip_files = sorted([zip_file for zip_file in os.listdir(zip_folder_path) if zip_file.endswith(".7z")])

    iteration = 0

    for zip_file in zip_files[first:last]:

        start_time = time.time()

        zip_file_index = zip_files.index(zip_file)

        Archive(zip_folder_path + "/" + zip_file).extractall(zip_folder_path)

        json_file_path = unzip_folder_path + "/" + zip_file.rstrip(".7z")

        output_file_path = dataset_path + "/" + zip_file.rstrip(".json.7z") + ".pkl"

        mapping_file_path = dataset_path + "/" + zip_file.rstrip(".json.7z") + ".mapping.pkl"

        with open(json_file_path, "r") as input_file, bz2.BZ2File(output_file_path, "wb") as output_file, bz2.BZ2File(mapping_file_path, "wb") as mapping_file:

            tables = list()

            mapping = dict()

            for line in input_file:

                raw_table = json.loads(line)

                table = dict()

                table["_id"] = str(zip_file_index) + "." + str(len(tables))

                table["entity"] = raw_table["key"]

                table["page"] = raw_table["pageID"]

                table["file"] = zip_file.rstrip(".json.7z") + ".pkl"

                if table["page"] not in mapping.keys():

                    mapping[table["page"]] = dict()

                if table["entity"] not in mapping[table["page"]].keys():

                    mapping[table["page"]][table["entity"]] = [table["_id"]]

                else:

                    mapping[table["page"]][table["entity"]].append(table["_id"])

                if "content" in raw_table.keys():

                    table["num_columns"] = raw_table["columns"]

                    table["num_rows"] = raw_table["rows"]

                    table["num_header_rows"] = "".join(re.findall(r"<t\w>", raw_table["content"])).count("<tr><th>")

                    table["context"] = [raw_table["pageTitle"], raw_table["headings"]]

                    table["content"] = raw_table["contentParsed"]

                if "validFrom" in raw_table.keys():

                    table["valid_from"] = raw_table["validFrom"]

                if "validTo" in raw_table.keys():

                    table["valid_to"] = raw_table["validTo"]

                table["revision"] = raw_table["revisionId"]

                tables.append(table)

            pickle.dump(tables, output_file)

            pickle.dump(mapping, mapping_file)

            input_file.close(), output_file.close(), mapping_file.close()

        end_time = time.time()

        print("Iteration " + str(iteration) + ": " + str(end_time - start_time) + " s")

        iteration += 1

        os.remove(json_file_path)

#To load in tables from pickles
def store_tables(first, last):

    dataset_path = "datasets/tobias_tables"

    client = MongoClient()

    db = client.blossom

    table_collection = db.tables

    iteration = 0

    table_files = sorted([table_file for table_file in os.listdir(dataset_path) if table_file.endswith(".output.pkl")])

    for file in table_files[first:last]:

        start_time = time.time()

        with bz2.BZ2File(dataset_path + "/" + file, "rb") as f:

            tables = pickle.load(f)

            if len(tables) > 0:

                table_collection.insert_many(tables)

            f.close()

        end_time = time.time()

        print("Iteration " + str(iteration) + ": " + str(end_time - start_time) + " s")

        iteration += 1