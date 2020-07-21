package BinaryTree;

/**
 * 定义二叉树可以实现的基本功能
 * @author Lenovo
 *
 */
public interface Tree {
	
	public int size();
	
	public boolean isEmpty();
	
	public int getHeight();
	
	public void preTraversal();
	
	public void middleTraversal();
	
	public void postTraversal();
	
	//借助队列实现层次遍历
	public void orderByQueue();
	
	//借助栈实现非递归遍历二叉树,使用先序进行遍历
	public void preTraByStack();
	//借助栈实现非递归遍历二叉树,使用中序进行遍历
	void inOrderByStack();
	//借助栈实现非递归遍历二叉树,使用后序进行遍历
	void postTraByStack();
}
