#pragma once
struct Node
{
	int x, y, u;
	int r = -1;	//result
	Node* westNode;

	Node() {}
	
	Node(int _x, int _y, int _u) : x(_x), y(_y), u(_u) {}

};
/*
struct Node{int x, y, u;int r;Node(){} Node* westNode;	Node(int _x, int _y, int _u) : x(_x), y(_y), u(_u) {} };
*/