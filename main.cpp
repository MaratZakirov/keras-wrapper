#include <iostream>
#include <Eigen/Dense>
#include <json/json.h>
#include <fstream>
#include <map>
#include <keras-wrapper.h>

using std::cout;
using std::endl;
using std::istringstream;
using std::getline;

typedef Eigen::MatrixXf Matrix;
typedef Eigen::VectorXf Vector;

void print(Vector v)
{
    cout << "print vector of size: " << v.size() << endl;
    cout << v << endl;
}

void print(Matrix m)
{
    cout << "print matrix of rows: " << m.rows() << " and cols: " << m.cols() << endl;
    cout << m << endl;
}

inline Vector sigmoid(const Vector& v)
{
    return (1.0 + (-v).array().exp()).inverse().matrix();
}

inline Vector hard_sigmoid(const Vector& v)
{
    return ((v.array() * 0.2) + 0.5).array().min(1.0).max(0.0);
}

inline Vector tanh(const Vector& v)
{
    auto exp = (-v).array().exp();
    auto exp_2 = exp * exp;
    return (1 - exp_2) / (1 + exp_2);
}

std::vector<std::wstring> split(const std::wstring &s, wchar_t delim)
{
    std::wstringstream ss(s);
    std::wstring item;
    std::vector<std::wstring> tokens;
    while (getline(ss, item, delim))
    {
        if (item.size() > 0)
            tokens.push_back(item);
    }
    return tokens;
}

std::vector<string> split(const string &s, char delim)
{
    std::stringstream ss(s);
    string item;
    std::vector<string> tokens;
    while (getline(ss, item, delim))
    {
        if (item.size() > 0)
            tokens.push_back(item);
    }
    return tokens;
}

Matrix matrixFromString(string & s)
{
    auto lines = split(s, '\n');
    int N = lines.size();
    int M = split(lines[0], ' ').size();
    Matrix m(N, M);
    for (int i = 0; i < N; i++)
    {
        auto cols = split(lines[i], ' ');
        for (int j = 0; j < M; j++)
            m(i, j) = std::stof(cols[j]);
    }
    return m.transpose().eval();
}

Vector vectorFromString(string & s)
{
    auto arr = split(s, ' ');
    Vector v(arr.size());
    for (unsigned i = 0; i < arr.size(); i++)
    {
        v(i) = std::stof(arr[i]);
    }
    return v;
}

class Layer
{
    protected:
    std::string type;
    std::string name;
    Layer() : type("NONE"), name("NONE"), ret_seq(true), next(NULL) {
        ;
    }
    public:
    bool ret_seq;
    Layer * next;
    virtual void Fill(Json::Value & root) = 0;
    virtual Vector Pass(Vector & input) = 0;
    virtual void Init() { ; };
};

class Dense : public Layer
{
    Matrix W;
    Vector b;
    public:
    Dense(int in_dim, int out_dim, std::string nm) : W(in_dim, out_dim), b(out_dim)
    {
        type = "dense";
        name = nm;
    }
    Vector Pass(Vector & input)
    {
        return W * input + b;
    }
    void Fill(Json::Value & root)
    {
        string W_s = root[name + "_W"].asCString();
        string b_s = root[name + "_b"].asCString();

        W = matrixFromString(W_s);
        b = vectorFromString(b_s);
    };
};

class Activation : public Layer
{
    std::string activation;
    public:
    Activation(std::string act, std::string nm) : activation(act)
    {
        type = "activation";
        name = nm;
    }
    Vector Pass(Vector & input)
    {
        if (activation == "linear")
            return input;
        else if (activation == "sigmoid")
            return sigmoid(input);
        else if (activation == "tanh")
            return tanh(input);
        else
            assert (0);
    }
    void Fill(Json::Value &)
    {
        ;
    }
};

class LSTM : public Layer
{
    Vector c_t_1;
    Vector h_t_1;

    public:
    Matrix W_f;
    Matrix U_f;
    Vector b_f;

    Matrix W_i;
    Matrix U_i;
    Vector b_i;

    Matrix W_o;
    Matrix U_o;
    Vector b_o;

    Matrix W_c;
    Matrix U_c;
    Vector b_c;

    LSTM(int in_dim, int out_dim, std::string nm, bool seq) :
            W_f(in_dim, out_dim), U_f(out_dim, out_dim), b_f(out_dim),
            W_i(in_dim, out_dim), U_i(out_dim, out_dim), b_i(out_dim),
            W_o(in_dim, out_dim), U_o(out_dim, out_dim), b_o(out_dim),
            W_c(in_dim, out_dim), U_c(out_dim, out_dim), b_c(out_dim),
            c_t_1(out_dim), h_t_1(out_dim)
    {
        type = "LSTM";
        name = nm;
        ret_seq = seq;
    }

    void Fill(Json::Value & root)
    {
        string W_f_s = root[name + "_W_f"].asCString();
        string U_f_s = root[name + "_U_f"].asCString();
        string b_f_s = root[name + "_b_f"].asCString();

        string W_i_s = root[name + "_W_i"].asCString();
        string U_i_s = root[name + "_U_i"].asCString();
        string b_i_s = root[name + "_b_i"].asCString();

        string W_o_s = root[name + "_W_o"].asCString();
        string U_o_s = root[name + "_U_o"].asCString();
        string b_o_s = root[name + "_b_o"].asCString();

        string W_c_s = root[name + "_W_c"].asCString();
        string U_c_s = root[name + "_U_c"].asCString();
        string b_c_s = root[name + "_b_c"].asCString();

        W_f = matrixFromString(W_f_s);
        U_f = matrixFromString(U_f_s);
        b_f = vectorFromString(b_f_s);

        W_i = matrixFromString(W_i_s);
        U_i = matrixFromString(U_i_s);
        b_i = vectorFromString(b_i_s);

        W_o = matrixFromString(W_o_s);
        U_o = matrixFromString(U_o_s);
        b_o = vectorFromString(b_o_s);

        W_c = matrixFromString(W_c_s);
        U_c = matrixFromString(U_c_s);
        b_c = vectorFromString(b_c_s);
    };

    void Init() {
        h_t_1.array() = 0;
        c_t_1.array() = 0;
    }

    Vector Pass(Vector & x) {
        Vector f_t = hard_sigmoid(W_f * x + U_f * h_t_1 + b_f);
        Vector i_t = hard_sigmoid(W_i * x + U_i * h_t_1 + b_i);
        Vector o_t = hard_sigmoid(W_o * x + U_o * h_t_1 + b_o);
        Vector c_t = f_t.array() * c_t_1.array() + i_t.array() * tanh(W_c * x + U_c * h_t_1 + b_c).array();
        Vector h_t = o_t.array() * tanh(c_t.array());
        h_t_1 = h_t;
        c_t_1 = c_t;
        return h_t;
    }
};

class Model
{
    public:
    Layer * first = NULL;
    Model(string & rank_file) {
        Json::Value root;
        Json::Reader reader_model;
        std::ifstream ifs_model(rank_file.c_str());
        Layer *prev_layer = NULL;
        std::map<std::string, Layer *> layer_map;

        reader_model.parse(ifs_model, root);

        auto config = root["struct"]["config"];

        // First load network structure
        for (int i = 0; i < config.size(); i++) {
            Layer *new_layer = NULL;
            auto elem = config[i];
            std::string layer_name = elem["config"]["name"].asCString();
            std::string layer_type_name = elem["class_name"].asCString();
            if (layer_type_name == "Dense") {
                int input_dim = elem["config"]["input_dim"].asInt();
                int output_dim = elem["config"]["output_dim"].asInt();

                new_layer = new Dense(input_dim, output_dim, layer_name);
            } else if (layer_type_name == "Activation") {
                std::string activation = elem["config"]["activation"].asCString();

                new_layer = new Activation(activation, layer_name);
            } else if (layer_type_name == "LSTM")
            {
                int input_dim = elem["config"]["input_dim"].asInt();
                int output_dim = elem["config"]["output_dim"].asInt();

                new_layer = new LSTM(input_dim, output_dim, layer_name, false);
            }
            // This done for masking. TODO: support masking in future explicitly
            if (new_layer) {
                layer_map[layer_name] = new_layer;
                if (first == NULL)
                    first = new_layer;
                if (prev_layer)
                    prev_layer->next = new_layer;
                prev_layer = new_layer;
            }
        }
        auto weights = root["weights"];
        for (auto const &map_elem : layer_map) {
            std::string layer_name = map_elem.first;
            Layer *layer = map_elem.second;
            layer->Fill(weights[layer_name]);
        }
    }
    std::vector<Vector> Pass(std::vector<Vector> & input) {
        Layer * layer = first;
        std::vector<Vector> result = input;
        while (layer) {
            std::vector<Vector> new_result;
            layer->Init();
            for (int i = 0; i < result.size(); i++) {
                auto vec_res = layer->Pass(result[i]);
                if (layer->ret_seq or i == (result.size() - 1))
                    new_result.push_back(vec_res);
            }
            result = new_result;
            layer = layer->next;
        }
        return result;
    }
};

std::vector<Vector> loadData(string & data_file) {
    std::vector<Vector> data;
    std::ifstream ifs_data(data_file.c_str());
    string line;
    while (getline(ifs_data, line)) {
        if (line == "")
            continue;
        auto elems = split(line, ' ');
        Vector input(elems.size());
        for (int i = 0; i < elems.size(); i++)
            input(i) = std::stof(elems[i]);
        data.push_back(input);
    }
    return data;
}

#ifdef NLP_PROC
class NLP
{
    std::map<std::wstring, int> bigram2id;
    std::wstring empty = L"";
    std::wstring resh  = L"#";
    public:
    NLP(string & bigram_file)
    {
        std::wifstream ifs_bigram(bigram_file.c_str());
        std::wstring line;
        int id = 0;
        while(getline(ifs_bigram, line))
        {
            if (line == empty)
                continue;
            std::wstring e = split(line, '\t')[0];
            bigram2id[e] = id;
            id++;
        }
    }
    std::vector<Vector> strToData(std::wstring & line)
    {
        std::wcout << line << endl;

        std::vector<Vector> result;
        std::vector<std::wstring> words = split(line, ' ');
        for (std::wstring word : words) {
            Vector v(bigram2id.size());
            std::wcout << word << endl;
            word = L"#" + word + L"#";
            std::wcout << word << endl;
            for (int i = 0; i < word.size() - 1; i++){
                std::wstring e = word.substr(i, 2);
                std::wcout << e << endl;
                if (bigram2id.find(e) != bigram2id.end())
                {
                    v(bigram2id[e]) += 1.0;
                }
            }
            if (v.sum() != 0)
                result.push_back(v);
        }
        return result;
    }
};
#endif

#if TEST
int main()
{
    string model_file = "/home/zakirov/proj/semantic/impl/model-structure.json";
    string param_file = "/home/zakirov/proj/semantic/impl/model-weights.json";
    string data_file = "/home/zakirov/proj/semantic/impl/data.txt";

    auto model = Model(model_file, param_file);

    Vector input(5);

    auto data = loadData(data_file);

    cout << "Perform" << endl;
    auto result = model.Pass(data);
    print(result[0]);
}
#endif

kerasModel::kerasModel(string rank_file)
    : model(std::make_shared<Model>(rank_file)) {
    ;
}

std::vector<float> kerasModel::pass(std::vector<std::vector<float>> &data) const {
    std::vector<Vector> data_2;
    for (int i = 0; i < data.size(); ++i)
    {
        Vector v(data[i].size());
        for (int j = 0; j < data[i].size(); ++j)
        {
            v(j) = data[i][j];
        }
        data_2.push_back(v);
    }
    Vector res_vec = model->Pass(data_2)[0];
    std::vector<float> res(res_vec.data(), res_vec.data() + res_vec.rows() * res_vec.cols());
    return res;
}
