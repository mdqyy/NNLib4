#ifndef IOXML_H
#define IOXML_H

#include <string>
#include <memory>
#include "RapidXML-1.13\rapidxml.hpp"
#include "RapidXML-1.13\rapidxml_print.hpp"
#include "NN.h"
#include "IOTreeNode.h"
#include <fstream>
#include <sstream>

class IOXML
{
	static void save_node(IOTreeNode& node, rapidxml::xml_node<>& xml_node, rapidxml::xml_document<>& doc)
	{
		IOTreeNodeData<std::string>& node_attributes = node.attributes();
		for (auto& iter = node_attributes.begin(); iter != node_attributes.end(); iter++)
		{
			char *attr_name = doc.allocate_string((*iter).c_str());
			char *attr_value = doc.allocate_string(node_attributes.GetEntry(*iter).c_str());
			rapidxml::xml_attribute<> *attr = doc.allocate_attribute( attr_name, attr_value );
			xml_node.append_attribute(attr);
		}
		
		IOTreeNodeData< std::shared_ptr<IOTreeNode> >& node_nodes = node.nodes();
		for (auto& iter = node.nodes().begin(); iter != node.nodes().end(); iter++)
		{
			char *node_name = doc.allocate_string((*iter).c_str());
			rapidxml::xml_node<> *inner_node = doc.allocate_node( rapidxml::node_element, node_name );
			xml_node.append_node(inner_node);
			save_node( *node.nodes().GetEntry( *iter ), *inner_node, doc);
		}
	}
	
	static std::shared_ptr< IOTreeNode > load_node(rapidxml::xml_node<>& xml_node)
	{
		std::shared_ptr< IOTreeNode > node( new IOTreeNode() );
		for (rapidxml::xml_attribute<> *attr = xml_node.first_attribute(); attr; attr = attr->next_attribute())
			node->attributes().AppendEntry( attr->name(), attr->value() );
		
		for (rapidxml::xml_node<> *inner_node = xml_node.first_node(); inner_node; inner_node = inner_node->next_sibling())
			node->nodes().AppendEntry( inner_node->name(), load_node(*inner_node) );

		return node;
	}

public:
	static void save(IOTreeNode& node, std::ostream& output_stream)
	{
		rapidxml::xml_document<> doc;
		char* main_node_name = "NN";
		rapidxml::xml_node<> *xml_node = doc.allocate_node( rapidxml::node_element, main_node_name );
		doc.append_node(xml_node);
		save_node(node, *xml_node, doc);

		using namespace rapidxml;
		output_stream<<doc;
	}

	static std::shared_ptr< IOTreeNode > load(std::istream& input_stream)
	{
		std::vector<char> data_text;
		data_text.assign( (std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>() );
		data_text.push_back('\0');

		rapidxml::xml_document<> doc;
		doc.parse<0>(data_text.data());
		rapidxml::xml_node<> *xml_node = doc.first_node();
		return load_node( *xml_node);
	}
};

#endif