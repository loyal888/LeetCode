package BinaryTree;

import BinaryTree.LinkedBinaryTree;
import BinaryTree.Node;

public class TestTree {
	
	public static void main(String[] args) {
		
		//创建一棵基本的二叉树
		Node node7  = new Node(7, null, null);
		Node node6  = new Node(6, null, null);
		Node node3  = new Node(3, null, null);
		Node node5  = new Node(5, node6, node7);
		Node node2  = new Node(2, node3, node5);
		Node node8  = new Node(8, null, null);
		Node node11  = new Node(11, null, null);
		Node node12 = new Node(12, null, null);
		Node node10  = new Node(10, node11, node12);
		Node node9  = new Node(9, node10, null);
		Node node4  = new Node(4, node8, node9);
		
		Node root  = new Node(1, node4, node2);
//		BinaryTree.LinkedBinaryTree link = new BinaryTree.LinkedBinaryTree();
		LinkedBinaryTree link = new LinkedBinaryTree(root);
		
		//查看树是否为空
		System.out.println(link.isEmpty());
		
		//前序递归遍历
		link.preTraversal();
		
		//中序递归遍历
		link.middleTraversal();
		
		//后序递归遍历
		link.postTraversal();
		
		//计算结点的个数
		int size = link.size();
		System.out.println("个数是："+size);
		//得到树的高度
		int height = link.getHeight();
		System.out.println("树的高度是："+height);
		//借助队列实现层次遍历
		link.orderByQueue();
		
		//借助栈实现中序遍历，不采用递归
		link.inOrderByStack();
		
		//借助栈实现先序遍历
		link.preTraByStack();
		System.out.println();
		//借助栈实现后续遍历
		link.postTraByStack();
		System.out.println();
	}
}
