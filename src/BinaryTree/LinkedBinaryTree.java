package BinaryTree;

import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.Queue;

/**
 * 采用双链表法实现二叉树
 * @author Lenovo
 */
public class LinkedBinaryTree  implements Tree {
	//为此棵数，创建根节点
	private Node root;
	
	public LinkedBinaryTree(Node root) {
		super();
		this.root = root;
	}
	
	public LinkedBinaryTree() {
		super();
	}
 
	@Override
	public int size() {
		return this.size(root);
	}
 
	private int size(Node node) {
		if(node==null){
			return 0;
		}else{
			int le = size(node.leftChild);
			int ri = size(node.rightChild);
			return ri+le+1;
		}
	}
 
	@Override
	public boolean isEmpty() {
		return root == null;
	}
 
	@Override
	public int getHeight() {
		return this.getHeight(root);
	}
	
	private int getHeight(Node node) {
		if(node == null){
			return 0;
		}else{
			int ai = this.getHeight(node.leftChild);
			int bi = this.getHeight(node.rightChild);
			return ai>bi?ai+1:bi+1;
		}
	}
 
	@Override
	public void preTraversal() {
		//进步封装完善方法
		//首先必须考虑代码健壮性
		if(root !=null){
			System.out.println("先序遍历：");
			pre(root);
			System.out.println();
		}else{
			System.out.println("此二叉树为空");
		}
	}

	/**
	 * 先序遍历(Pre-order),按照根左右的顺序沿一定路径经过路径上所有的结点。
	 * @param node
	 */
	private void pre(Node node) {
		if(node!=null){
		//输出二叉树根节点
		System.out.print(node.data+"  ");
		//继续相应地遍历左子树和右子树	
		this.pre(node.leftChild);
		this.pre(node.rightChild);
		}
	}
 
	@Override
	public void middleTraversal() {
		//进步封装完善方法
		//首先必须考虑代码健壮性
		if(root !=null){
			System.out.println("中序遍历：");
			middle(root);
			System.out.println();
		}else{
			System.out.println("此二叉树为空");
		}
	}

	/**
	 * 中序遍历首先遍历左子树，然后访问根结点，最后遍历右子树。
	 * @param node
	 */
	private void middle(Node node) {
		if(node!=null){
		//输出二叉树根节点
		this.middle(node.leftChild);
		System.out.print(node.data+"  ");
		this.middle(node.rightChild);
		}
	}
 
	@Override
	public void postTraversal() {
		//进步封装完善方法
		//首先必须考虑代码健壮性
		if(root !=null){
			System.out.println("中序遍历：");
			pos(root);
			System.out.println();
		}else{
			System.out.println("此二叉树为空");
		}
	}
 
	private void pos(Node node) {
		if(node!=null){
		//输出二叉树根节点
		this.pos(node.leftChild);
		this.pos(node.rightChild);
		System.out.print(node.data+"  ");
		}	
	}
 
	@Override
	public void orderByQueue() {
		//新建一个队列，存放二叉树结点
		Queue qu = new LinkedList<>();
		Node node = root;
		qu.add(node);
		//遍历输出每一层，同时将下一层的结点加入到对列中,队列现有的个数，就是每层节点的个数
		while(qu.size()!=0){
			for(int i = 0;i<qu.size();i++)
			{
				Node po = (Node) qu.poll();
				System.out.print(po.data+"  ");
				if(po.leftChild != null)  
					qu.add(po.leftChild);
				if(po.rightChild != null)  
					qu.add(po.rightChild);
			}
		}
		System.out.println();
	}
 
	@Override
	public void preTraByStack() {
		System.out.println("不采用递归，实现先序遍历，借助栈");
		//新建栈
		Deque<Node> de = new LinkedList<>();
		Node node = root;
		de.push(node);
		while(node != null&&de.size()>0){
			node = de.pop();
			System.out.print(node.data+"  ");
			if(node.leftChild!=null && node.rightChild!=null){
				de.push(node.rightChild);
				de.push(node.leftChild);
			}else if(node.leftChild!=null &&node.rightChild == null){
				de.push(node.leftChild);
			}else if(node.leftChild ==null &&node.rightChild != null){
				de.push(node.rightChild);
			}else{
				/*BinaryTree.Node pop = de.pop();
				System.out.print(pop.data+"  ");*/
			}
		}
	}
	
	public void postTree(){
		System.out.println("借助外部方法实现后续遍历：");
		Deque<Node> stack =new LinkedList<Node>();
		Node node = root;
		Node proot;//标记栈顶元素前一个被访问的元素
		int flag;//root的左孩子未被访问；
		if(node!=null){
			do{
				while(node!=null){//将root所有左孩子全部入栈
					stack.push(root);
					node=node.leftChild;
				  }
				
				//执行到此处，栈顶元素没有左孩子或者左子树已经被访问过；
				proot=null;//标记栈顶元素前一个被访问的元素，或者此时为最左下边，该元素前一个被访问的元素肯定为空。
				flag=1;//root的左孩子已经被访问；或者root为null
				
				while(!stack.isEmpty() && flag==1){
					node=stack.peek();       //取到栈顶元素，但是不出栈；
					if(node.rightChild==proot){
						node=stack.pop();
						System.out.print(node.data+"  ");
						proot=node;
					}else{
						node=node.rightChild;
						flag=0;//root左边孩子未被访问；
					}
				}
			}while(!stack.isEmpty());
		}
	}
 
	@Override
	public void postTraByStack() {
		System.out.println("后序非递归遍历，借助栈");
		//新建栈，先进后出,将根结点入栈,双端队列
		Deque<Node> stack = new LinkedList<>();
		//新建一个list，记录结点的状态是否已经被访问过
		ArrayList<Node> list = new ArrayList<>();
//		stack.push(root);
		Node proot;
		Node node = root;
		int flag;
		//首先检查完树的左子树，再右子树，最后将根节点输出
		while(node != null  || stack.size()>0){
			//将最左子树添加完毕
			while(node != null){
				stack.push(node);
				node = node.leftChild;
			}
			//和中序遍历相似，为先输出左结点，但是做结点输出完毕之后，不能直接将根结点弹出，而是必须先将右结点弹出，
			//最后再将根结点弹出来，就会牵扯到一个根结点的访问状态的问题，是否已经被遍历过了
			//利用一个list集合记录已将被遍历过的根结点，防止产生死循环
			if(stack.size()> 0 ){
				Node peek = stack.peek();
				if(peek.rightChild!=null){
					boolean con = list.contains(peek);
					if(con ==true){
						Node pop = stack.pop();
						System.out.print(pop.data+"  ");
					}else{
						list.add(peek);
						node = peek.rightChild;
					}
				}else{
					Node pop = stack.pop();
					System.out.print(pop.data+"  ");
				}
			}
		}
	}
	
	@Override
	public void inOrderByStack() {
		System.out.println("中序非递归遍历:");
		// 创建栈,和先序遍历类似，直接入栈直到没有最左左子树可以 入栈
		Deque<Node> stack = new LinkedList<Node>();
		Node node = root;
		//添加暂时完毕，开始pop元素
		while(node!=null || stack.size()>0 ){
			
			while(node!=null){
			stack.push(node);
			node = node.leftChild;
			}
			//一边pop并且一边进行判断，右结点不会null的，右子树，继续按照添加方法，将最左结点全部添加进去
			if(stack.size()>0){
				Node pop = stack.pop();
				System.out.print(pop.data+"  ");
				if(pop.rightChild!=null){
					node = pop.rightChild;
				}
			}
		}
		System.out.println();
	}
}
