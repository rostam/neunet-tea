//
// Created by rostam on 11.11.19.
//

#ifndef CNEUNET_TEA_CSV_H
#define CNEUNET_TEA_CSV_H

#include <vector>
#include <sstream>

namespace neunet {

//    std::vector<std::string> split(const std::string &line, char delimiter) {
//        std::vector<std::string> tokens;
//        std::string token;
//        std::istringstream tokenStream(line);
//        while (std::getline(tokenStream, token, delimiter)) {
//            tokens.push_back(token);
//        }
//        return tokens;
//    }
//
//    void read_csv(std::string csv_file) {
//        std::ifstream file(root_csv_map);
//        std::string str;
//        while (std::getline(file, str)) {
//            std::vector<std::string> tokens = split(str, ' ');
//            std::vector<std::string> splitnames = split(tokens.at(1), '_');
//
//            std::string name_w_spaces;
//            for (auto i: splitnames) name_w_spaces = name_w_spaces + i + " ";
//
//            people_names.insert(std::make_pair(stoi(tokens.at(0)), name_w_spaces));
//            people_images.insert(std::make_pair(stoi(tokens.at(0)), std::string("database/images/" + tokens.at(2))));
//
//        }
//    }
}

#endif //CNEUNET_TEA_CSV_H
