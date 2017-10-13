#include <memory>
#include <string>
#include <vector>

using std::string;

class Model;
class kerasModel
{
    std::shared_ptr<Model> model;
public:
    kerasModel(string rank_file);
    std::vector<float> pass(std::vector<std::vector<float>> & data) const;
};
