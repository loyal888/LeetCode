
import java.util.*;

public class Solution {


    /**
     * 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，
     * 每一列都按照从上到下递增的顺序排序。
     * 请完成一个函数，输入这样的一个二维数组和一个整数
     * ，判断数组中是否含有该整数。
     *
     * @param target
     * @param array
     * @return
     */
    public boolean Find(int target, int[][] array) {
        // array 为空
        if (array.length == 0) {
            return false;
        }
        // 获取行数
        int row = array.length;
        // 获取列数
        int col = array[0].length;
        // 一直循环
        int i = 0, j = col - 1;
        while (i < row && j >= 0) {
            if (array[i][j] > target) {
                j--;
            } else if (array[i][j] < target) {
                i++;
            } else {
                return true;
            }
        }
        return false;
    }


    /**
     * 请实现一个函数，将一个字符串中的每个空格替换成“%20”。
     * 例如，当字符串为We Are Happy.
     * 则经过替换之后的字符串为We%20Are%20Happy。
     *
     * @param str
     * @return
     */
    public String replaceSpace(StringBuffer str) {
        return str.toString().replace(" ", "%20");
    }

    public static class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    /**
     * 输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
     *
     * @param listNode
     * @return
     */
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> arrayList = new ArrayList<Integer>();
        // 传入的是空节点
        if (listNode == null) {
            return arrayList;
        }
        ListNode L = null;
        ListNode M = null;
        ListNode R = listNode;
        // 先反转链表
        while (R.next != null) {
            L = M;
            M = R;
            R = R.next;
            M.next = L;
        }
        R.next = M;
        // 遍历取值
        while (R.next != null) {
            arrayList.add(R.val);
            R = R.next;
        }
        arrayList.add(R.val);
        return arrayList;
    }
    // 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    public void push(int node) {
        stack1.push(node);

    }


    public int pop() {
        int data = 0;
        // 先把stack1中的值搬到stack2中
        while (!stack1.empty()) {
            stack2.push(stack1.pop());
        }
        // 取出栈顶数据
        data = stack2.pop();
        // stack2 中的数据放回
        while (!stack2.empty()) {
            stack1.add(stack2.pop());
        }
        // 返回data
        return data;
    }

    /**
     * 大家都知道斐波那契数列，现在要求输入一个整数n，
     * 请你输出斐波那契数列的第n项（从0开始，第0项为0）。
     * n<=39
     *
     * @param n 斐波那契数列的第n项
     * @return
     */
    public int Fibonacci(int n) {
        if (n < 0) {
            System.out.print("n can not 小于 0");
            return 0;
        }
        if (n == 0) {
            return 0;
        }
        if (n == 1 || n == 2) {
            return 1;
        }
        return Fibonacci(n - 1) + Fibonacci(n - 2);

    }

    /**
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级。
     * 求该青蛙跳上一个n级的台阶总共有多少种跳法
     * （先后次序不同算不同的结果）。
     * <p>
     * 解: 第一次跳一级时，f(1)= 1;余下f(n-1)
     * 第一次跳二级时，f(2)= 1;余下f(n-2)
     * f(n) = f(n-1)+f(n-2)
     *
     * @param target
     * @return
     */
    public int JumpFloor(int target) {
        if (target < 0) {
            System.out.print("target can not 小于 0");
            return 0;
        }
        if (target == 0) {
            return 0;
        }
        if (target == 1) {
            return 1;
        }
        if (target == 2) {
            return 2;
        }
        return JumpFloor(target - 1) + JumpFloor(target - 2);

    }

    /**
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级……
     * 它也可以跳上n级。
     * 求该青蛙跳上一个n级的台阶总共有多少种跳法。
     * <p>
     * 解： 关于本题，前提是n个台阶会有一次n阶的跳法。分析如下:
     * f(1) = 1
     * f(2) = f(2-1) + f(2-2)         //f(2-2) 表示2阶一次跳2阶的次数。
     * f(3) = f(3-1) + f(3-2) + f(3-3)..
     * f(n) = f(n-1) + f(n-2) + f(n-3) + ... + f(n-(n-1)) + f(n-n)
     * 说明：
     * 1）这里的f(n) 代表的是n个台阶有一次1,2,...n阶的 跳法数。
     * 2）n = 1时，只有1种跳法，f(1) = 1
     * 3) n = 2时，会有两个跳得方式，一次1阶或者2阶，这回归到了问题（1） ，f(2) = f(2-1) + f(2-2)
     * 4) n = 3时，会有三种跳得方式，1阶、2阶、3阶，
     * 那么就是第一次跳出1阶后面剩下：f(3-1);第一次跳出2阶，剩下f(3-2)；第一次3阶，那么剩下f(3-3)
     * 因此结论是f(3) = f(3-1)+f(3-2)+f(3-3)
     * 5) n = n时，会有n中跳的方式，1阶、2阶...n阶，得出结论：
     * f(n) = f(n-1)+f(n-2)+...+f(n-(n-1)) + f(n-n) => f(0) + f(1) + f(2) + f(3) + ... + f(n-1)
     * 6) 由以上已经是一种结论，但是为了简单，我们可以继续简化：
     * f(n-1) = f(0) + f(1)+f(2)+f(3) + ... + f((n-1)-1) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2)
     * f(n) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2) + f(n-1) = f(n-1) + f(n-1)
     * 可以得出：
     * f(n) = 2*f(n-1)
     * 7) 得出最终结论,在n阶台阶，一次有1、2、...n阶的跳的方式时，总得跳法为：
     * <p>
     * | 1       ,(n=0 )
     * <p>
     * f(n) =     | 1       ,(n=1 )
     * <p>
     * | 2*f(n-1),(n>=2)
     *
     * @param target
     * @return
     */
    public int JumpFloorII(int target) {
        if (target <= 0) {
            return -1;
        } else if (target == 1) {
            return 1;
        } else {
            return 2 * JumpFloorII(target - 1);
        }
    }


    /**
     * 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。
     * 请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，
     * 总共有多少种方法？
     * 解：
     * 依旧是斐波那契数列
     * 2*n的大矩形，和n个2*1的小矩形
     * 其中target*2为大矩阵的大小
     * 有以下几种情形：
     * 1⃣️target <= 0 大矩形为<= 2*0,直接return 1；
     * 2⃣️target = 1大矩形为2*1，只有一种摆放方法，return1；
     * 3⃣️target = 2 大矩形为2*2，有两种摆放方法，return2；
     * 4⃣️target = n 分为两步考虑：
     * 第一次摆放一块 2*1 的小矩阵，则摆放方法总共为f(target - 1)
     * √
     * √
     * 第一次摆放一块1*2的小矩阵，则摆放方法总共为f(target-2)
     * 因为，摆放了一块1*2的小矩阵（用√√表示），对应下方的1*2（用××表示）摆放方法就确定了，所以为f(targte-2)
     * √	√
     * ×	×
     *
     * @param target
     * @return
     */
    public int RectCover(int target) {
        if (target <= 1) {
            return 1;
        }
        if (target * 2 == 2) {
            return 1;
        } else if (target * 2 == 4) {
            return 2;
        } else {
            return RectCover((target - 1)) + RectCover(target - 2);
        }
    }

    /**
     * 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
     * 解：
     * 如果一个整数不为0，那么这个整数至少有一位是1。如果我们把这个整数减1，那么原来处在整数最右边的1就会变为0，
     * 原来在1后面的所有的0都会变成1(如果最右边的1后面还有0的话)。其余所有位将不会受到影响。
     * 举个例子：一个二进制数1100，从右边数起第三位是处于最右边的一个1。减去1后，第三位变成0，
     * 它后面的两位0变成了1，而前面的1保持不变，因此得到的结果是1011.
     * 我们发现减1的结果是把最右边的一个1开始的所有位都取反了。
     * 这个时候如果我们再把原来的整数和减去1之后的结果做与运算，
     * 从原来整数最右边一个1那一位开始所有位都会变成0。
     * 如1100&1011=1000.也就是说，把一个整数减去1，再和原整数做与运算，
     * 会把该整数最右边一个1变成0.那么一个整数的二进制有多少个1，就可以进行多少次这样的操作。
     *
     * @param n
     * @return
     */
    public int NumberOf1(int n) {
        int count = 0;
        while (n != 0) {
            count++;
            n = n & (n - 1);
        }
        return count;
    }

    /**
     * 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
     * 保证base和exponent不同时为0
     * <p>
     * 1.全面考察指数的正负、底数是否为零等情况。
     * 2.写出指数的二进制表达，例如13表达为二进制1101。
     * 3.举例:10^1101 = 10^0001*10^0100*10^1000。
     * 4.通过&1和>>1来逐位读取1101，为1时将该位代表的乘数累乘到最终结果。
     */
    public double Power(double base, int n) {
        double res = 1, curr = base;
        int exponent;
        if (n > 0) {
            exponent = n;
        } else if (n < 0) {
            if (base == 0) {
                throw new RuntimeException("分母不能为0");
            }
            exponent = -n;
        } else {// n==0
            return 1;// 0的0次方
        }
        while (exponent != 0) {
            if ((exponent & 1) == 1) {
                res *= curr;
            }
            curr *= curr;// 翻倍
            exponent >>= 1;// 右移一位
        }
        return n >= 0 ? res : (1 / res);
    }


    /**
     * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
     * 使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
     * 并保证奇数和奇数，偶数和偶数之间的相对位置不变。
     *
     * @param array
     */
    public void reOrderArray(int[] array) {
        //相对位置不变，稳定性
        //插入排序的思想
        // 数组长度
        int length = array.length;
        // 用来记录奇数的个数
        int k = 0;
        for (int i = 0; i < length; i++) {
            int j = i;
            if (array[i] % 2 == 1) {
                while (j > k) {
                    int temp = array[j];
                    array[j] = array[j - 1];
                    array[j - 1] = temp;
                    j--;
                }
                k++;
            }
        }
    }

    /**
     * 输入一个链表，输出该链表中倒数第k个结点。
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null) {
            return head;
        }
        int length = 1;
        ListNode p = head;
        while (p.next != null) {
            length++;
            p = p.next;
        }
        if (length < k) {
            return null;
        }
        int i = 1;
        while (i < length - k + 1) {
            head = head.next;
            i++;
        }
        return head;
    }

    /**
     * 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
     *
     * @param list1
     * @param list2
     * @return
     */
    public ListNode Merge(ListNode list1, ListNode list2) {
        // 为空的情况
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }
        // 结点不空
        ListNode node = null;
        if (list1.val > list2.val) {
            node = list2;
            node.next = Merge(list1, list2.next);
        } else {
            node = list1;
            node.next = Merge(list1.next, list2);
        }
        return node;
    }

    public static class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int val) {
            this.val = val;

        }

    }

    /**
     * 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
     *
     * @param root1
     * @param root2
     * @return
     */
    public static boolean HasSubtree(TreeNode root1, TreeNode root2) {
        boolean result = false;
        //当Tree1和Tree2都不为零的时候，才进行比较。否则直接返回false
        if (root2 != null && root1 != null) {
            //如果找到了对应Tree2的根节点的点
            if (root1.val == root2.val) {
                //以这个根节点为为起点判断是否包含Tree2
                result = doesTree1HaveTree2(root1, root2);
            }
            //如果找不到，那么就再去root的左儿子当作起点，去判断时候包含Tree2
            if (!result) {
                result = HasSubtree(root1.left, root2);
            }

            //如果还找不到，那么就再去root的右儿子当作起点，去判断时候包含Tree2
            if (!result) {
                result = HasSubtree(root1.right, root2);
            }
        }
        //返回结果
        return result;
    }

    public static boolean doesTree1HaveTree2(TreeNode node1, TreeNode node2) {
        //如果Tree2已经遍历完了都能对应的上，返回true
        if (node2 == null) {
            return true;
        }
        //如果Tree2还没有遍历完，Tree1却遍历完了。返回false
        if (node1 == null) {
            return false;
        }
        //如果其中有一个点没有对应上，返回false
        if (node1.val != node2.val) {
            return false;
        }

        //如果根节点对应的上，那么就分别去子节点里面匹配
        return doesTree1HaveTree2(node1.left, node2.left) && doesTree1HaveTree2(node1.right, node2.right);
    }

    /**
     * 操作给定的二叉树，将其变换为源二叉树的镜像。
     * 二叉树的镜像定义：源二叉树
     * 8
     * /  \
     * 6   10
     * / \  / \
     * 5  7 9 11
     * 镜像二叉树
     * 8
     * /  \
     * 10   6
     * / \  / \
     * 11 9 7  5
     * 先前序遍历这棵树的每个结点，如果遍历到的结点有子结点，就交换它的两个子节点，
     * 当交换完所有的非叶子结点的左右子结点之后，就得到了树的镜像
     *
     * @param root
     */
    public void Mirror(TreeNode root) {
        if (root == null) {
            return;
        }
        if (root.left == null && root.right == null) {
            return;
        }

        TreeNode pTemp = root.left;
        root.left = root.right;
        root.right = pTemp;

        if (root.left != null) {
            Mirror(root.left);
        }
        if (root.right != null) {
            Mirror(root.right);
        }

    }


    /**
     * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
     * 假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，
     * 序列4,5,3,2,1是该压栈序列对应的一个弹出序列，
     * 但4,3,5,1,2就不可能是该压栈序列的弹出序列。
     * （注意：这两个序列的长度是相等的）
     *
     * @param pushA
     * @param popA
     * @return
     */
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA.length == 0 || popA.length == 0) {
            return false;
        }
        ;

        Stack<Integer> stack = new Stack<>();
        int popIndex = 0;
        for (int i = 0; i < pushA.length; i++) {
            stack.push(pushA[i]);
            //如果栈不为空，且栈顶元素等于弹出序列
            while (!stack.empty() && stack.peek() == popA[popIndex]) {
                //出栈
                stack.pop();
                //弹出序列向后一位
                popIndex++;
            }
        }
        return stack.empty();
    }

    /**
     * 从上往下打印出二叉树的每个节点，同层节点从左至右打印。
     * <p>
     * 用arraylist模拟一个队列来存储相应的TreeNode
     *
     * @param root
     * @return
     */
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        ArrayList<TreeNode> queue = new ArrayList<>();
        if (root == null) {
            return list;
        }
        queue.add(root);
        while (queue.size() != 0) {
            TreeNode temp = queue.remove(0);
            if (temp.left != null) {
                queue.add(temp.left);
            }
            if (temp.right != null) {
                queue.add(temp.right);
            }
            list.add(temp.val);
        }
        return list;
    }

    /**
     * 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
     * 如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
     * （后序遍历：首先遍历左子树，然后遍历右子树，最后访问根结点，
     * 在遍历左、右子树时，仍然先遍历左子树，然后遍历右子树，最后遍历根结点）
     * <p>
     * 解：BST的后序序列的合法序列是，对于一个序列S，最后一个元素是x
     * （也就是根），如果去掉最后一个元素的序列为T，
     * 那么T满足：T可以分成两段，前一段（左子树）小于x，后一段（右子树）大于x，
     * 且这两段（子树）都是合法的后序序列。完美的递归定义 : )
     *
     * @param sequence
     * @return
     */
    public boolean VerifySquenceOfBST(int[] sequence) {
        int count = sequence.length;
        if (count == 0) {
            return false;
        }
        return isRight(sequence, 0, count - 1);

    }

    public boolean isRight(int[] sequence, int start, int end) {
        if (start >= end) {
            return true;
        }
        int i = end - 1;
        while (sequence[i] > sequence[end] && i > start) {
            i--;
        }
        for (int j = start; j < i; j++) {
            if (sequence[j] > sequence[end]) {
                return false;
            }
        }
        return isRight(sequence, start, i) && isRight(sequence, i + 1, end - 1);


    }

    /**
     * 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
     * 例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
     * 由于数字2在数组中出现了5次，超过数组长度的一半，
     * 因此输出2。如果不存在则输出0。
     *
     * @param array
     * @return
     */
    public int MoreThanHalfNum_Solution(int[] array) {
        // 测试用例 [1,2,2]->2; [1,2,3]->0; []->0;  [1]->1; [1,2]->0;
        if (array.length == 0) {
            return 0;
        }
        if (array.length == 1) {
            return array[0];
        }
        // 遍历数组 记录出现的次数
        HashMap<Integer, Integer> count = new HashMap<>();
        for (int i = 0; i < array.length; i++) {
            if (!count.containsKey(array[i])) {
                count.put(array[i], 1);
            } else {
                Integer value = count.get(array[i]);
                count.put(array[i], value + 1);
            }
        }
        // 判断次数是否超过数组长度的一半
        for (Integer i : count.keySet()) {
            if (count.get(i) > array.length / 2) {
                return i;
            }
        }
        return 0;
    }

    /**
     * 快排实现
     *
     * @param num
     * @param left
     * @param right
     * @return
     */

    private static void QuickSort(int[] num, int left, int right) {
        //如果left等于right，即数组只有一个元素，直接返回
        if (left >= right) {
            return;
        }
        //设置最左边的元素为基准值
        int key = num[left];
        //数组中比key小的放在左边，比key大的放在右边，key值下标为i
        int i = left;
        int j = right;
        while (i < j) {
            //j向左移，直到遇到比key小的值
            while (num[j] >= key && i < j) {
                j--;
            }
            //i向右移，直到遇到比key大的值
            while (num[i] <= key && i < j) {
                i++;
            }
            //i和j指向的元素交换
            if (i < j) {
                int temp = num[i];
                num[i] = num[j];
                num[j] = temp;
            }
        }
        num[left] = num[i];
        num[i] = key;
        QuickSort(num, left, i - 1);
        QuickSort(num, i + 1, right);
    }

    public static void merge(int[] a, int low, int mid, int high) {
        int[] temp = new int[high - low + 1];
        int i = low;// 左指针
        int j = mid + 1;// 右指针
        int k = 0;
        // 把较小的数先移到新数组中
        while (i <= mid && j <= high) {
            if (a[i] < a[j]) {
                temp[k++] = a[i++];
            } else {
                temp[k++] = a[j++];
            }
        }
        // 把左边剩余的数移入数组
        while (i <= mid) {
            temp[k++] = a[i++];
        }
        // 把右边边剩余的数移入数组
        while (j <= high) {
            temp[k++] = a[j++];
        }
        // 把新数组中的数覆盖nums数组
        for (int k2 = 0; k2 < temp.length; k2++) {
            a[k2 + low] = temp[k2];
        }
    }

    /**
     * 归并排序
     *
     * @param a
     * @param low
     * @param high
     */
    public static void mergeSort(int[] a, int low, int high) {
        int mid = (low + high) / 2;
        if (low < high) {
            // 左边
            mergeSort(a, low, mid);
            // 右边
            mergeSort(a, mid + 1, high);
            // 左右归并
            merge(a, low, mid, high);
        }

    }

    /**
     * 时间复杂度O(n)，空间复杂度O(1）
     * 查找数组中重复的数字
     * 在长度为n的数组中，所有的元素都是0到n-1的范围内。
     * 数组中的某些数字是重复的，但不知道有几个重复的数字，
     * 也不知道重复了几次，请找出任意重复的数字。
     * 例如，输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出为2或3
     *
     * @param nums
     * @return
     */
    public static ArrayList<Integer> findDuplicatedNum(int[] nums) {
        ArrayList<Integer> integers = new ArrayList<>();
        if (nums == null || nums.length <= 0) {
            return null;
        }
        // 数据不在0-n-1区间
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] < 0 || nums[i] > nums.length - 1) {
                return null;
            }
        }
        // 数组中的数字为 0 到 n-1 的范围内。如果这个数组中没有重复的数字，则对应的 i 位置的数据也为 i。
        // 可以重排此数组，扫描数组中的每一个数字，当扫描到下标为 i 的数字时，首先比较这个数字（m）是不是等于 i。
        // 如果是，接着扫描下一个数字。如果不是，再拿它和第m 个数字比较，
        // 如果相等则找到重复的数据。否则就把第 i 个数字与第 m 个数字交换。
        // 重复这个比较、交换的过程，直到找到重复的数字。
        for (int i = 0; i < nums.length; ++i) {
            while (nums[i] != i) {
                if (nums[i] == nums[nums[i]]) {
                    integers.add(nums[i]);
                    // 找到该数字后 就结束循环
                    break;
                }
                // 交换数字
                int temp = nums[i];
                nums[i] = nums[temp];
                nums[temp] = temp;
            }
        }

        return integers;
    }

    /**
     * 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
     * 暴力法
     * // TODO 学习非暴力解决办法
     *
     * @param input
     * @param k
     * @return
     */
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> minKs = new ArrayList<>();
        if (k > input.length) {
            return minKs;
        }
        QuickSort(input, 0, input.length - 1);
        for (int i = 0; i < k; i++) {
            minKs.add(input[i]);
        }
        return minKs;
    }

    /**
     * 暴力法
     * 题目描述:
     * HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。
     * 今天测试组开完会后,他又发话了:在古老的一维模式识别中
     * 常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。
     * 但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？
     * 例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。
     * 给一个数组，返回它的最大连续子序列的和，
     * 你会不会被他忽悠住？(子向量的长度至少是1)
     *
     * @param array
     * @return
     */
    public static int FindGreatestSumOfSubArray(int[] array) {
        ArrayList<Integer> integers = new ArrayList<>();
        for (int i = 0; i < array.length; i++) {
            integers.add(array[i]);
            int sum = array[i];
            for (int j = i + 1; j < array.length; j++) {
                sum += array[j];
                integers.add(sum);
            }
        }
        Collections.sort(integers);
        return integers.get(integers.size() - 1);
    }

    /**
     * 使用动态规划
     * F（i）：以array[i]为末尾元素的子数组的和的最大值，子数组的元素的相对位置不变
     * F（i）=max（F（i-1）+array[i] ， array[i]）
     * res：所有子数组的和的最大值
     * res=max（res，F（i））
     * <p>
     * 如数组[6, -3, -2, 7, -15, 1, 2, 2]
     * 初始状态：
     * F（0）=6
     * res=6
     * i=1：
     * F（1）=max（F（0）-3，-3）=max（6-3，3）=3
     * res=max（F（1），res）=max（3，6）=6
     * i=2：
     * F（2）=max（F（1）-2，-2）=max（3-2，-2）=1
     * res=max（F（2），res）=max（1，6）=6
     * i=3：
     * F（3）=max（F（2）+7，7）=max（1+7，7）=8
     * res=max（F（3），res）=max（8，6）=8
     * i=4：
     * F（4）=max（F（3）-15，-15）=max（8-15，-15）=-7
     * res=max（F（4），res）=max（-7，8）=8
     * 以此类推
     * 最终res的值为8     * @param array
     *
     * @return
     */
    public static int FindGreatestSumOfSubArray1(int[] array) {
        // 记录当前所有子数组的和的最大值
        int res = array[0];
        // 包含array[i]的连续数组最大值
        int max = array[0];
        for (int i = 1; i < array.length; i++) {
            max = Math.max(max + array[i], array[i]);
            res = Math.max(max, res);
        }
        return res;
    }


    /**
     * 暴力法
     * 求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？
     * 为此他特别数了一下1~13中包含1的数字有1、10、11、12、13
     * 因此共出现6次,但是对于后面问题他就没辙了。
     * ACMer希望你们帮帮他,并把问题更加普遍化
     * 可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。
     *
     * @param n
     * @return
     */
    public static int NumberOf1Between1AndN_Solution(int n) {
        int count = 0;
        for (int i = 0; i <= n; i++) {
            Integer integer = i;
            char one = '1';
            String s = integer.toString();
            for (int j = 0; j < s.length(); j++) {
                if (one == s.charAt(j)) {
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
     * 例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
     *
     * @param numbers
     * @return 解题思路：
     * 先将整型数组转换成String数组，然后将String数组排序，最后将排好序的字符串数组拼接出来。关键就是制定排序规则。
     * 排序规则如下：
     * 若ab > ba 则 a > b，
     * 若ab < ba 则 a < b，
     * 若ab = ba 则 a = b；
     * 解释说明：
     * 比如 "3" < "31"但是 "331" > "313"，所以要将二者拼接起来进行比较
     */
    public static String PrintMinNumber(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return "";
        }
        int len = numbers.length;
        String[] str = new String[len];
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < len; i++) {
            str[i] = String.valueOf(numbers[i]);
        }
        Arrays.sort(str, new Comparator<String>() {
            @Override
            public int compare(String s1, String s2) {
                String c1 = s1 + s2;
                String c2 = s2 + s1;
                return c1.compareTo(c2);
            }
        });
        for (int i = 0; i < len; i++) {
            sb.append(str[i]);
        }
        return sb.toString();
    }

    /**
     * 把只包含质因子2、3和5的数称作丑数（Ugly Number）。
     * 例如6、8都是丑数，但14不是，因为它包含质因子7。
     * 习惯上我们把1当做是第一个丑数。
     * 求按从小到大的顺序的第N个丑数。
     * <p>
     * <p>
     * 通俗易懂的解释：
     * 首先从丑数的定义我们知道，一个丑数的因子只有2,3,5，那么丑数p = 2 ^ x * 3 ^ y * 5 ^ z，换句话说一个丑数一定由另一个丑数乘以2或者乘以3或者乘以5得到，那么我们从1开始乘以2,3,5，就得到2,3,5三个丑数，在从这三个丑数出发乘以2,3,5就得到4，6,10,6，9,15,10,15,25九个丑数，我们发现这种方法会得到重复的丑数，而且我们题目要求第N个丑数，这样的方法得到的丑数也是无序的。那么我们可以维护三个队列：
     * （1）丑数数组： 1
     * 乘以2的队列：2
     * 乘以3的队列：3
     * 乘以5的队列：5
     * 选择三个队列头最小的数2加入丑数数组，同时将该最小的数乘以2,3,5放入三个队列；
     * （2）丑数数组：1,2
     * 乘以2的队列：4
     * 乘以3的队列：3，6
     * 乘以5的队列：5，10
     * 选择三个队列头最小的数3加入丑数数组，同时将该最小的数乘以2,3,5放入三个队列；
     * （3）丑数数组：1,2,3
     * 乘以2的队列：4,6
     * 乘以3的队列：6,9
     * 乘以5的队列：5,10,15
     * 选择三个队列头里最小的数4加入丑数数组，同时将该最小的数乘以2,3,5放入三个队列；
     * （4）丑数数组：1,2,3,4
     * 乘以2的队列：6，8
     * 乘以3的队列：6,9,12
     * 乘以5的队列：5,10,15,20
     * 选择三个队列头里最小的数5加入丑数数组，同时将该最小的数乘以2,3,5放入三个队列；
     * （5）丑数数组：1,2,3,4,5
     * 乘以2的队列：6,8,10，
     * 乘以3的队列：6,9,12,15
     * 乘以5的队列：10,15,20,25
     * 选择三个队列头里最小的数6加入丑数数组，但我们发现，有两个队列头都为6，所以我们弹出两个队列头，同时将12,18,30放入三个队列；
     * ……………………
     * 疑问：
     * 1.为什么分三个队列？
     * 丑数数组里的数一定是有序的，因为我们是从丑数数组里的数乘以2,3,5选出的最小数，一定比以前未乘以2,3,5大，同时对于三个队列内部，按先后顺序乘以2,3,5分别放入，所以同一个队列内部也是有序的；
     * 2.为什么比较三个队列头部最小的数放入丑数数组？
     * 因为三个队列是有序的，所以取出三个头中最小的，等同于找到了三个队列所有数中最小的。
     * 实现思路：
     * 我们没有必要维护三个队列，只需要记录三个指针显示到达哪一步；“|”表示指针,arr表示丑数数组；
     * （1）1
     * |2
     * |3
     * |5
     * 目前指针指向0,0,0，队列头arr[0] * 2 = 2,  arr[0] * 3 = 3,  arr[0] * 5 = 5
     * （2）1 2
     * 2 |4
     * |3 6
     * |5 10
     * 目前指针指向1,0,0，队列头arr[1] * 2 = 4,  arr[0] * 3 = 3, arr[0] * 5 = 5
     * （3）1 2 3
     * 2| 4 6
     * 3 |6 9
     * |5 10 15
     * 目前指针指向1,1,0，队列头arr[1] * 2 = 4,  arr[1] * 3 = 6, arr[0] * 5 = 5
     *
     * @param index
     * @return
     */
    public static int GetUglyNumber_Solution(int index) {
        if (index < 7) {
            return index;
        }
        ArrayList<Integer> arr = new ArrayList<>();
        //p2，p3，p5分别为三个队列的指针，newNum为从队列头选出来的最小数
        int p2 = 0, p3 = 0, p5 = 0, newNum = 1;
        arr.add(newNum);
        while (arr.size() < index) {
            //选出三个队列头最小的数
            newNum = Math.min(arr.get(p2) * 2, Math.min(arr.get(p3) * 3, arr.get(p5) * 5));
            //这三个if有可能进入一个或者多个，进入多个是三个队列头最小的数有多个的情况
            if (arr.get(p2) * 2 == newNum) {
                p2++;
            }
            ;
            if (arr.get(p3) * 3 == newNum) {
                p3++;
            }
            if (arr.get(p5) * 5 == newNum) {
                p5++;
            }
            arr.add(newNum);
        }
        return arr.get(index - 1);
    }

    /**
     * @param str
     * @return 说一下解题思路哈，其实主要还是hash，
     * 利用每个字母的ASCII码作hash来作为数组的index。
     * 首先用一个58长度的数组来存储每个字母出现的次数，
     * 为什么是58呢，主要是由于A-Z对应的ASCII码为65-90，
     * a-z对应的ASCII码值为97-122，而每个字母的index=int(word)-65，
     * 比如g=103-65=38，而数组中具体记录的内容是该字母出现的次数，
     * 最终遍历一遍字符串，找出第一个数组内容为1的字母就可以了，
     * 时间复杂度为O(n)
     */
    public static int FirstNotRepeatingChar(String str) {
        int[] words = new int[58];
        for (int i = 0; i < str.length(); i++) {
            words[((int) str.charAt(i)) - 65] += 1;
        }
        for (int i = 0; i < str.length(); i++) {
            if (words[((int) str.charAt(i)) - 65] == 1) {
                return i;
            }
        }
        return -1;
    }

    /**
     * 归并排序的改进，把数据分成前后两个数组(递归分到每个数组仅有一个数据项)，
     * 合并数组，合并时，出现前面的数组值array[i]大于后面数组值array[j]时；则前面
     * 数组array[i]~array[mid]都是大于array[j]的，count += mid+1 - i
     * 参考剑指Offer，但是感觉剑指Offer归并过程少了一步拷贝过程。
     * 还有就是测试用例输出结果比较大，对每次返回的count mod(1000000007)求余
     *
     * @param array
     * @return
     */
    public int InversePairs(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int[] copy = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            copy[i] = array[i];
        }
        //数值过大求余
        int count = InversePairsCore(array, copy, 0, array.length - 1);
        return count;

    }

    private int InversePairsCore(int[] array, int[] copy, int low, int high) {
        if (low == high) {
            return 0;
        }
        int mid = (low + high) >> 1;
        int leftCount = InversePairsCore(array, copy, low, mid) % 1000000007;
        int rightCount = InversePairsCore(array, copy, mid + 1, high) % 1000000007;
        int count = 0;
        int i = mid;
        int j = high;
        int locCopy = high;
        while (i >= low && j > mid) {
            if (array[i] > array[j]) {
                count += j - mid;
                copy[locCopy--] = array[i--];
                // 数值过大求余
                if (count >= 1000000007) {
                    count %= 1000000007;
                }
            } else {
                copy[locCopy--] = array[j--];
            }
        }
        for (; i >= low; i--) {
            copy[locCopy--] = array[i];
        }
        for (; j > mid; j--) {
            copy[locCopy--] = array[j];
        }
        for (int s = low; s <= high; s++) {
            array[s] = copy[s];
        }
        return (leftCount + rightCount + count) % 1000000007;
    }

    /**
     * 输入两个链表，找出它们的第一个公共结点。
     * 例如： 1->3->5->6 2->3->5
     * 5是公共结点
     *
     * @param pHead1
     * @param pHead2
     * @return
     */
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        ListNode p1 = pHead1;
        ListNode p2 = pHead2;
        while (p1 != p2) {
            p1 = (p1 == null ? pHead2 : p1.next);
            p2 = (p2 == null ? pHead1 : p2.next);
        }
        return p1;
    }


    /**
     * 统计一个数字在排序数组中出现的次数。
     *
     * @param array
     * @param k
     * @return
     */
    public int GetNumberOfK(int[] array, int k) {
        int count = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] == k) {
                count++;
            }
        }
        return count;
    }

    /**
     * 输入一棵二叉树，求该树的深度。
     * 从根结点到叶结点依次经过的结点（含根、叶结点）
     * 形成树的一条路径，最长路径的长度为树的深度。
     * 递归层次遍历
     *
     * @param root
     * @return
     */
    public int TreeDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = TreeDepth(root.left);
        int right = TreeDepth(root.right);
        return Math.max(left, right) + 1;
    }

    /**
     * 输入一棵二叉树，判断该二叉树是否是平衡二叉树。
     * 平衡二叉树（Balanced Binary Tree）具有以下性质：
     * 它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，
     * 并且左右两个子树都是一棵平衡二叉树。
     *
     * @param root
     * @return
     */
    public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null) {
            return true;
        }
        return Math.abs(maxDepth(root.left) - maxDepth(root.right)) <= 1 &&
                IsBalanced_Solution(root.left) && IsBalanced_Solution(root.right);
    }

    private int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    /**
     * 一个整型数组里除了两个数字之外，其他的数字都出现了两次。
     * 请写程序找出这两个只出现一次的数字。
     *
     * @param array
     * @param num1
     * @param num2
     */
    public static void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {
        if (array == null) {
            num1 = null;
            num2 = null;
        }
        HashMap<Integer, Integer> arrayMap = new HashMap<>();
        for (int num : array) {
            if (arrayMap.containsKey(num)) {
                arrayMap.put(num, (arrayMap.get(num)) + 1);
            } else {
                arrayMap.put(num, 1);
            }
        }
        int[] num = new int[2];
        int j = 0;
        for (int anArray : array) {
            Integer integer = arrayMap.get(anArray);
            if (integer == 1) {
                num[j] = anArray;
                j++;
            }
        }
        num1[0] = num[0];
        num2[0] = num[1];
    }

    /**
     * 小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。
     * 但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。
     * 没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。
     * 现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列?
     * Good Luck!
     * <p>
     * 1）由于我们要找的是和为S的连续正数序列，因此这个序列是个公差为1的等差数列，
     * 而这个序列的中间值代表了平均值的大小。假设序列长度为n，
     * 那么这个序列的中间值可以通过（S / n）得到，知道序列的中间值和长度，
     * 也就不难求出这段序列了。
     * 2）满足条件的n分两种情况：
     * n为奇数时，序列中间的数正好是序列的平均值，所以条件为：(n & 1) == 1 && sum % n == 0；
     * n为偶数时，序列中间两个数的平均值是序列的平均值，而这个平均值的小数部分为0.5，
     * 所以条件为：(sum % n) * 2 == n.
     * 3）由题可知n >= 2，那么n的最大值是多少呢？我们完全可以将n从2到S全部遍历一次，
     * 但是大部分遍历是不必要的。为了让n尽可能大，我们让序列从1开始，
     * 根据等差数列的求和公式：S = (1 + n) * n / 2，得到. n < 根号（2*s）
     * 最后举一个例子，假设输入sum = 100，我们只需遍历n = 13~2的情况（按题意应从大到小遍历），n = 8时，得到序列[9, 10, 11, 12, 13, 14, 15, 16]；n  = 5时，得到序列[18, 19, 20, 21, 22]。
     * 完整代码：时间复杂度为
     *
     * @param sum
     * @return
     */
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> ans = new ArrayList<>();
        for (int n = (int) Math.sqrt(2 * sum); n >= 2; n--) {
            if ((n & 1) == 1 && sum % n == 0 || (sum % n) * 2 == n) {
                ArrayList<Integer> list = new ArrayList<>();
                for (int j = 0, k = (sum / n) - (n - 1) / 2; j < n; j++, k++) {
                    list.add(k);
                }
                ans.add(list);
            }
        }
        return ans;
    }

    /**
     * 输入一个递增排序的数组和一个数字S，在数组中查找两个数，
     * 使得他们的和正好是S，如果有多对数字的和等于S
     * ，输出两个数的乘积最小的。
     *
     * @param array
     * @param sum
     * @return
     */
    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
        ArrayList<Integer> integers = new ArrayList<>();
        boolean find = false;
        for (int i = 0; i < array.length; i++) {
            for (int j = array.length - 1; j >= i; j--) {
                if (array[i] + array[j] == sum) {
                    integers.add(array[i]);
                    integers.add(array[j]);
                    find = true;
                }
            }
            if (find) {
                break;
            }
        }
        return integers;
    }

    /**
     * 汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。
     *
     * @param str
     * @param n
     * @return
     */
    public String LeftRotateString(String str, int n) {
        if (n > str.length()) {
            return "";
        }
        return new String(str.getBytes(), n, str.length() - n) + new String(str.getBytes(), 0, n);
    }

    /**
     * 牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。
     * 同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。
     * 例如，“student. a am I”。
     * 后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。
     * Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？
     * @param args
     */
    /**
     * 牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。
     * 同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。
     * 例如，“student. a am I”。
     * 后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。
     * Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？
     */
    public static String ReverseSentence(String str) {
        if (" ".equals(str)) {
            return " ";
        }
        Stack<String> chs = new Stack<>();
        String[] strs = str.split(" ");
        chs.addAll(Arrays.asList(strs));
        StringBuilder s = new StringBuilder();
        while (!chs.empty()) {
            s.append(chs.pop());
            s.append(" ");
        }
        return s.toString().trim();
    }

    /**
     * 每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。
     * HF作为牛客的资深元老,自然也准备了一些小游戏。
     * 其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。
     * 然后,他随机指定一个数m,让编号为0的小朋友开始报数。
     * 每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中
     * 从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,
     * 可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。
     * 请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
     * 如果没有小朋友，请返回-1
     * <p>
     * 约瑟夫环 x'=(x+k)%n
     */
    int LastRemaining_Solution(int n, int m) {
        if (n == 0) {
            return -1;
        }
        if (n == 1) {
            return 0;
        } else {
            return (LastRemaining_Solution(n - 1, m) + m) % n;
        }
    }

    /**
     * 求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、
     * case等关键字及条件判断语句（A?B:C）。
     *
     * @param n
     * @return
     */
    public int Sum_Solution(int n) {
        int sum = n;
        boolean ans = (n > 0) && ((sum += Sum_Solution(n - 1)) > 0);
        return sum;
    }

    /**
     * 整数相加
     *
     * @param num1
     * @param num2
     * @return
     */
    public static int Add(int num1, int num2) {
        while (num2 != 0) {
            int temp = num1 ^ num2;
            num2 = (num1 & num2) << 1;
            num1 = temp;
        }
        return num1;
    }

    public static boolean flag;

    /**
     * 将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能
     * ，但是string不符合数字要求时返回0)，
     * 要求不能使用字符串转换整数的库函数。
     * 数值为0或者字符串不是一个合法的数值则返回0
     *
     * @param str
     * @return
     */
    public static int StrToInt(String str) {
        flag = false;
        //判断输入是否合法
        if (str == null || str.trim().equals("")) {
            flag = true;
            return 0;
        }
        // symbol=0,说明该数为正数;symbol=1，该数为负数;start用来区分第一位是否为符号位
        int symbol = 0;
        int start = 0;
        char[] chars = str.trim().toCharArray();
        if (chars[0] == '+') {
            start = 1;
        } else if (chars[0] == '-') {
            start = 1;
            symbol = 1;
        }
        int result = 0;
        for (int i = start; i < chars.length; i++) {
            if (chars[i] > '9' || chars[i] < '0') {
                flag = true;
                return 0;
            }
            int sum = result * 10 + (int) (chars[i] - '0');


            if ((sum - (int) (chars[i] - '0')) / 10 != result) {
                flag = true;
                return 0;
            }

            result = result * 10 + (int) (chars[i] - '0');
            /*
             * 本人认为java热门第一判断是否溢出是错误的，举个反例
             * 当输入为value=2147483648时，在计算机内部的表示应该是-2147483648
             * 显然value>Integer.MAX_VALUE是不成立的
             */
        }
        // 注意：java中-1的n次方不能用：(-1)^n .'^'异或运算
        // 注意，当value=-2147483648时，value=-value
        result = (int) Math.pow(-1, symbol) * result;
        return result;
    }

    /**
     * 在一个长度为n的数组里的所有数字都在0到n-1的范围内。
     * 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
     *
     * @param numbers
     * @param length
     * @param duplication
     * @return
     */
    public boolean duplicate(int numbers[], int length, int[] duplication) {
        boolean flag = false;
        for (int i = 0; i < length; i++) {
            if (flag) {
                break;
            }
            for (int j = i + 1; j < length; j++) {
                if (numbers[j] == numbers[i]) {
                    duplication[0] = numbers[i];
                    flag = true;
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 因为值都小于n-1 故可以用一个boolean数组保存
     * k[numbers[i]] == true 就说明前面有这个数字了
     *
     * @param numbers
     * @param length
     * @param duplication
     * @return
     */
    public static boolean duplicate1(int numbers[], int length, int[] duplication) {
        boolean[] k = new boolean[length];
        for (int i = 0; i < k.length; i++) {
            if (k[numbers[i]] == true) {
                duplication[0] = numbers[i];
                return true;
            }
            k[numbers[i]] = true;
        }
        return false;
    }

    /**
     * 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],
     * 其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。
     * 不能使用除法。
     *
     * @param A
     * @return
     */
    public static int[] multiply(int[] A) {
        int length = A.length;
        int[] B = new int[length];
        if (length != 0) {
            B[0] = 1;
            //计算下三角连乘
            for (int i = 1; i < length; i++) {
                B[i] = B[i - 1] * A[i - 1];
            }
            int temp = 1;
            //计算上三角
            for (int j = length - 2; j >= 0; j--) {
                temp *= A[j + 1];
                B[j] *= temp;
            }
        }
        return B;
    }

    /**
     * 请实现一个函数用来匹配包括'.'和'*'的正则表达式。
     * 模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。
     * 在本题中，匹配是指字符串的所有字符匹配整个模式。
     * 例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
     *
     * @param str
     * @param pattern
     * @return
     */
    public boolean match(char[] str, char[] pattern) {
        return false;
    }

    /**
     * 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
     * 例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。
     * 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
     *
     * @param str
     * @return
     */
    public boolean isNumeric(char[] str) {
        String string = String.valueOf(str);
        return string.matches("[\\+\\-]?\\d*(\\.\\d+)?([Ee][\\+\\-]?\\d+)?");
    }

    public ListNode EntryNodeOfLoop(ListNode pHead) {
        ArrayList<ListNode> nodes = new ArrayList<>();
        while (pHead.next != null) {
            if (nodes.contains(pHead.next)) {
                return pHead.next;
            }
            if (!nodes.contains(pHead)) {
                nodes.add(pHead);
            }
            pHead = pHead.next;

        }
        return null;
    }

    /**
     * 断链法
     *
     * @param pHead
     * @return
     */
    public ListNode EntryNodeOfLoop2(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return null;
        }
        ListNode fast = pHead.next;
        ListNode slow = pHead;
        while (fast != null) {
            slow.next = null;
            slow = fast;
            fast = fast.next;
        }
        return slow;
    }

    //    /**
//     * 这段代码可以去除重复的链表结点并保留一个
//     * @param pHead
//     * @return
//     */
//    public static ListNode deleteDuplication(ListNode pHead) {
//        ListNode p = pHead;
//        while (pHead.next != null) {
//            if (pHead.val == pHead.next.val) {
//                ListNode pre = pHead;
//                while (pre.val == pHead.next.val) {
//                    pHead = pHead.next;
//                }
//                pre.next = pHead.next;
//            }
//            pHead = pHead.next;
//        }
//        return p;
//    }

    /**
     * 思路：首先根节点以及其左右子树，左子树的左子树和右子树的右子树相同
     * * 左子树的右子树和右子树的左子树相同即可，采用递归
     * * 非递归也可，采用栈或队列存取各级子树根节点
     *
     * @return
     */
    boolean isSymmetrical(TreeNode pRoot) {
        if (pRoot == null) {
            return true;
        }
        return comRoot(pRoot.left, pRoot.right);
    }

    private boolean comRoot(TreeNode left, TreeNode right) {
        if (left == null) {
            return right == null;
        }
        if (right == null) {
            return false;
        }
        if (left.val != right.val) {
            return false;
        }
        return comRoot(left.right, right.left) && comRoot(left.left, right.right);
    }

    /**
     * 大家的实现很多都是将每层的数据存进ArrayList中，偶数层时进行reverse操作，
     * 在海量数据时，这样效率太低了。
     * （我有一次面试，算法考的就是之字形打印二叉树，用了reverse，
     * 直接被鄙视了，面试官说海量数据时效率根本就不行。）
     * <p>
     * 下面的实现：不必将每层的数据存进ArrayList中，偶数层时进行reverse操作，直接按打印顺序存入
     * 思路：利用Java中的LinkedList的底层实现是双向链表的特点。
     * 1)可用做队列,实现树的层次遍历
     * 2)可双向遍历,奇数层时从前向后遍历，偶数层时从后向前遍历
     */
    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        if (pRoot == null) {
            return ret;
        }
        ArrayList<Integer> list = new ArrayList<>();
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.addLast(null);//层分隔符
        queue.addLast(pRoot);
        boolean leftToRight = true;

        while (queue.size() != 1) {
            TreeNode node = queue.removeFirst();
            if (node == null) {//到达层分隔符
                Iterator<TreeNode> iter = null;
                if (leftToRight) {
                    iter = queue.iterator();//从前往后遍历
                } else {
                    iter = queue.descendingIterator();//从后往前遍历
                }
                leftToRight = !leftToRight;
                while (iter.hasNext()) {
                    TreeNode temp = (TreeNode) iter.next();
                    list.add(temp.val);
                }
                ret.add(new ArrayList<Integer>(list));
                list.clear();
                queue.addLast(null);//添加层分隔符
                continue;//一定要continue
            }
            if (node.left != null) {
                queue.addLast(node.left);
            }
            if (node.right != null) {
                queue.addLast(node.right);
            }
        }

        return ret;
    }

    /**
     * 请实现两个函数，分别用来序列化和反序列化二叉树
     * <p>
     * 二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串
     * 从而使得内存中建立起来的二叉树可以持久保存。
     * 序列化可以基于先序、中序、后序、层序的二叉树遍历方式来进行修改，
     * 序列化的结果是一个字符串，序列化时通过 某种符号表示空节点（#），
     * 以 ！ 表示一个结点值的结束（value!）。
     * <p>
     * 二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树
     */
    public int index = -1;

    String Serialize(TreeNode root) {
        StringBuffer sb = new StringBuffer();
        if (root == null) {
            sb.append("#,");
            return sb.toString();
        }
        sb.append(root.val + ",");
        sb.append(Serialize(root.left));
        sb.append(Serialize(root.right));
        return sb.toString();
    }

    TreeNode Deserialize(String str) {
        index++;
        int len = str.length();
        if (index >= len) {
            return null;
        }
        String[] strr = str.split(",");
        TreeNode node = null;
        if (!strr[index].equals("#")) {
            node = new TreeNode(Integer.valueOf(strr[index]));
            node.left = Deserialize(str);
            node.right = Deserialize(str);
        }

        return node;
    }

    /**
     * 从左到右打印
     *
     * @param pRoot
     * @return
     */
    ArrayList<ArrayList<Integer>> Print1(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        if (pRoot == null) {
            return ret;
        }
        ArrayList<Integer> list = new ArrayList<>();
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.addLast(null);
        queue.addLast(pRoot);
        while (queue.size() != 1) {
            TreeNode node = queue.removeFirst();
            if (node == null) {
                Iterator<TreeNode> iterator = queue.iterator();
                while (iterator.hasNext()) {
                    TreeNode temp = (TreeNode) iterator.next();
                    list.add(temp.val);
                }
                ret.add(new ArrayList<Integer>(list));
                list.clear();
                queue.addLast(null);
                continue;
            } else {
                if (node.left != null) {
                    queue.addLast(node.left);
                }
                if (node.right != null) {
                    queue.addLast(node.right);
                }
            }

        }
        return ret;
    }

    /**
     * 给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。
     *
     * @param k
     * @return
     */
    static TreeNode KthNode(TreeNode root, int k) {
        int index = 0;
        if (root != null) { //中序遍历寻找第k个
            TreeNode node = KthNode(root.left, k);
            if (node != null) {
                return node;
            }
            index++;
            if (index == k) {
                return root;
            }
            node = KthNode(root.right, k);
            if (node != null) {
                return node;
            }
        }
        return null;
    }

    public static void main(String args[]) {
        TreeNode node = new TreeNode(5);
        TreeNode node1 = new TreeNode(3);
        TreeNode node2 = new TreeNode(7);
        node.left = node1;
        node.right = node2;

        KthNode(node, 2);
//        int[] array = new int[]{2, 7, 9};
//        twoSum(array, 9);
//        List<String> ans = new ArrayList();
//        backtrack(ans, "", 0, 0, 3);
//        convert("leetcod", 3);
        NumberOf1Between1AndN_Solution1(29, 33, '3');
    }

    public static int NumberOf1Between1AndN_Solution1(int m, int n, char k) {
        int count = 0;
        Scanner sc = new Scanner(System.in);

        for (int i = m; i <= n; i++) {
            Integer integer = i;
            char one = k;
            String s = integer.toString();
            for (int j = 0; j < s.length(); j++) {
                if (one == s.charAt(j)) {
                    count++;
                }
            }
        }
        return count;
    }
}