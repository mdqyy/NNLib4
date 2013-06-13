#ifndef IO_TREE_NODE_H
#define IO_TREE_NODE_H

#include <unordered_map>
#include <memory>
#include <string>

template <class T>
class IOTreeNodeData
{
	std::unordered_map<std::string, T> node_data_;
	std::vector<std::string> keys_in_append_order_;
public:
	
	typedef typename std::vector<std::string>::iterator Iterator;

	bool HasEntry(std::string name)
	{
		return node_data_.find(name) != node_data_.end();
	}
	
	T GetEntry(std::string name)
	{
		return node_data_[name];
	}

	void AppendEntry(std::string name, T data)
	{
		keys_in_append_order_.push_back(name);
		node_data_[name] = data;
	}

	void Clear()
	{
		node_data_.clear();
		keys_in_append_order_.clear();
	}

	Iterator begin()
	{
		return keys_in_append_order_.begin();
	}
	
	Iterator end()
	{
		return keys_in_append_order_.end();
	}
};

class IOTreeNode
{
	IOTreeNodeData<std::string> node_attributes_;
	IOTreeNodeData< std::shared_ptr<IOTreeNode> > inner_nodes_;
public:

	IOTreeNodeData<std::string>& attributes()
	{
		return node_attributes_;
	}
	IOTreeNodeData< std::shared_ptr<IOTreeNode> >& nodes()
	{
		return inner_nodes_;
	}

};

#endif